from openai import OpenAI
from dotenv import load_dotenv
import os
from rich import print as rprint
import ast
import re

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

class Executor:
    def __init__(self, agent, openai_tools, terminate_function):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        # Initialize cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # GPT-4 pricing per 1M tokens (as of March 2024)
        self.PROMPT_COST_PER_1M = 0.15
        self.COMPLETION_COST_PER_1M = 0.60

    def get_cost_info(self):
        total_cost = (self.total_prompt_tokens / 1000000.0 * self.PROMPT_COST_PER_1M) + \
                    (self.total_completion_tokens / 1000000.0 * self.COMPLETION_COST_PER_1M)
        return {
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
            'total_cost': total_cost
        }

    def step(self, observation: str):
        # Call the agent's tools or send messages
        self.agent.update_history({"role": "user", "content": observation})
        rprint(f"[bold magenta]\[`user` -> `{self.agent.name.lower()}`][/bold magenta]")
        rprint(f"[green]'{observation}'[/green]\n")
        response = self._call_llm()
        function_name, processed_response, tool_call_use = self._process_llm_response(response)
            
        while tool_call_use:
            if function_name == self.terminate_function:
                self.agent.update_history(processed_response)
                return True
            else:
                self.agent.update_history(processed_response)
                response = self._call_llm()
                function_name, processed_response, tool_call_use = self._process_llm_response(response)
        
        self.agent.update_history(processed_response)
        return False
    
    def _call_llm(self):
        # Call OpenAI model (or other models in the future)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
            messages=self.agent.history,
            tools=self.openai_tools,
            tool_choice="auto"  # Let the model decide if it should call the function
        )
        # Update token counts
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message
    
    def _process_llm_response(self, response):
        tool_call = response.tool_calls
        # Check if the assistant returned a function call
        if tool_call != None:
            tool_call_use = True
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

            if response.content != None:
                tool_result = [{
                    "role": "assistant",
                    "content": response.content
                }]
            else:
                tool_result = []
            tool_result.append({
                "role": "assistant",
                "content": f"tool call: {tool_name}, arguments: {arguments}"
            })
            tool_result.append({
                "role": "assistant",
                "content": f"tool response: {tool_name}, tool result: {tool_output}"
            })
            # Return the function result (so we can show it to the user, for example)
            return tool_name, tool_result, tool_call_use
            
        else:
            # If no function call, it's a normal text response
            tool_call_use = False
            tool_name = None
            assistant_content = response.content
            tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
            rprint(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            rprint(f"[green]'{assistant_content}'[/green]\n")
            if tool_name != None:
                rprint(f"[blue]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/blue]")
            return tool_name, {"role": "assistant", "content": assistant_content}, tool_call_use
        
    def _parse_content_for_tool_call(self, content: str):
        pattern = re.compile(r'\btool call:\s*(?P<func>(?:\w+\.)*(?P<clean_func>\w+))\s*,\s*arguments:\s*(?P<args>{.*})', re.DOTALL)
    
        match = pattern.search(content)
        if match:
            func_name = match.group('clean_func')  # Extract the last part of the function name
            args_str = match.group('args')
            
            try:
                # Safely parse the arguments using AST
                args_dict = ast.literal_eval(args_str)
                return func_name, args_dict
            except (SyntaxError, ValueError):
                return func_name, None
        
        return None, None
