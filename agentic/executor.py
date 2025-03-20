from openai import OpenAI
from dotenv import load_dotenv
import os
from rich import print as rprint

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

class Executor:
    def __init__(self, agent, openai_tools, terminate_function):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function

    def step(self, observation: str):
        # Call the agent's tools or send messages
        self.agent.update_history({"role": "user", "content": observation})
        rprint(f"[bold magenta]\[`user` -> `{self.agent.name.lower()}`][/bold magenta]")
        rprint(f"[green]'{observation}'[/green]\n")
        response = self._call_llm()
        function_name, processed_response = self._process_llm_response(response)
            
        while function_name != None:
            if function_name == self.terminate_function:
                self.agent.update_history(processed_response)
                return True
            else:
                self.agent.update_history(processed_response)
                response = self._call_llm()
                function_name, processed_response = self._process_llm_response(response)
        
        self.agent.update_history(processed_response)
        return False
    
    def _call_llm(self):
        # Call OpenAI model (or other models in the future)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=self.agent.history,
            tools=self.openai_tools,
            tool_choice="auto"  # Let the model decide if it should call the function
        )
        return response.choices[0].message
    
    def _process_llm_response(self, response):
        tool_call = response.tool_calls
        # Check if the assistant returned a function call
        if tool_call != None:
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            arguments = eval(tool_call.function.arguments)

            if tool_name not in self.agent.tools:
                raise ValueError(f"Function {tool_name} not found in tools")
            
            if tool_name == self.terminate_function:
                rprint(f"[bold red]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/bold red]\n")
                tool_output = None
            else:
                tool = self.agent.tools[tool_name]
                rprint(f"[blue]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/blue]")
                tool_output = tool(**arguments)
                rprint(f"[blue]\[tool response -->] {tool_output}[/blue]\n")

            tool_result = [{
                "role": "assistant",
                "content": f"tool call: {tool_name}, arguments: {arguments}"
            }, {
                "role": "assistant",
                "content": f"tool response: {tool_name}, tool result: {tool_output}"
            }]
            # Return the function result (so we can show it to the user, for example)
            return tool_name, tool_result
            
        else:
            # If no function call, it's a normal text response
            assistant_content = response.content
            rprint(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            rprint(f"[green]'{assistant_content}'[/green]\n")
            return None, {"role": "assistant", "content": assistant_content}

    def is_termination_condition_met(self, scenario):
        # Check if the scenario should terminate
        return len(self.history) >= len(scenario['system_messages'])