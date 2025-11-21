import os
import gradio as gr
from smolagents import (
    Tool,
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel
)

token = os.getenv("HF_TOKEN")

class SearchTool(Tool):
    name = "search_tool"
    description = "Search the web and return the most relevant info."
    inputs = {"query": {"type": "string", "description": "search query"}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.ddg = DuckDuckGoSearchTool()

    def forward(self, query: str):
        return self.ddg.run(query)

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides full self-contained answers."
    inputs = {"answer": {"type": "string", "description": "self-contained final answer"}}
    output_type = "string"

    def forward(self, answer: str):
        return f"Final Answer: {answer}"

class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Adds two numbers."
    inputs = {
        "a": {"type": "number", "description": "first param"},
        "b": {"type": "number", "description": "second param"},
    }
    output_type = "number"

    def forward(self, a, b):
        return a + b

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=token
)

prompt_templates={
    "system_prompt": (
        """
        You are the AI agent capable of using tools to solve user tasks.
        You have the following tools.
        Tool: AddNumbersTool.
        Tool: searchTool.
        Tool: FinalAnswerTool.
        Thought: I will use a tool to solve each seperate {task}.
        Final Answer: For each seperate {task} I will use the FinalAnswerTool to create full self-contained answers.
        """
    ),
    "planning": {"initial_plan": "", "update_plan_pre_messages": "", "update_plan_post_messages": ""},
    "final_answer": {"pre_messages": "", "post_messages": ""}, 
    "managed_agent": {"task": "", "report": ""},
}

agent = CodeAgent(
    model=model,
    tools=[
        AddNumbersTool(),
        SearchTool(),
        FinalAnswerTool(),
    ],
    max_steps=10,
    prompt_templates=prompt_templates
)

def run_agent(query: str):
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(lines=8, placeholder="Ask me anything..."),
    outputs=gr.Textbox(lines=8),
    title="SmolAgent",
    description=(
        "This agent uses DuckDuckGo search, adds numbers, and returns final answers"
    )
)

iface.launch(server_name="0.0.0.0", server_port=7860)