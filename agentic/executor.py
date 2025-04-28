from rich import print as rprint
from llm_client import LiteLlmClient
import time

class Executor:
    def __init__(self, agent, openai_tools, terminate_function, target_function, model_name):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        self.target_function = target_function
        self.model_name = model_name
        self.max_target_tries = 4
        self.max_retries = 5
        self.client = LiteLlmClient(model_name)
        # Initialize cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # GPT-4 pricing per 1M tokens (as of March 2024)
        self.PROMPT_COST_PER_1M = 1.10
        self.COMPLETION_COST_PER_1M = 4.40

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
        self.target_function_count = 0
        function_name, processed_response, retry_call = self._process_llm_response(response)
        if function_name == self.target_function:
            self.target_function_count += 1

        num_retries = 0
        while retry_call:
            if processed_response == None:
                response = self._call_llm()
                function_name, processed_response, retry_call = self._process_llm_response(response)
            elif function_name == self.terminate_function:
                self.agent.update_history(processed_response)
                return True
            else:
                self.agent.update_history(processed_response)
                response = self._call_llm()
                function_name, processed_response, retry_call = self._process_llm_response(response)
            num_retries += 1
            if num_retries > self.max_retries:
                break

        if processed_response != None:
            self.agent.update_history(processed_response)

        if function_name == self.terminate_function:
            return True
        return False
    
    def _call_llm(self):
        # start_time = time.time()
        response = self.client.get_completion(messages=self.agent.history, tools=self.openai_tools, tool_choice="auto")
        # end_time = time.time()
        # duration = end_time - start_time
        # rprint(f"[yellow]LLM call took {duration:.2f} seconds[/yellow]")
        # Update token counts
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message
    
    def _process_llm_response(self, response):
        tool_call = response.tool_calls
        # Check if the assistant returned a function call
        if tool_call != None:
            retry_call = True
            if len(response.tool_calls) > 1:
                return None, None, True
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            arguments = eval(tool_call.function.arguments)

            if tool_name not in self.agent.tools:
                raise ValueError(f"Function {tool_name} not found in tools")
            
            if tool_name == self.terminate_function:
                rprint(f"[bold red]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/bold red]\n")
                tool_output = None
                retry_call = False
            else:
                tool = self.agent.tools[tool_name]
                rprint(f"[blue]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/blue]")
                tool_output = tool(**arguments)
                if tool_name == self.target_function:
                    self.target_function_count += 1
                if self.target_function_count > self.max_target_tries:
                    tool_output = f"Maximum number of {tool_name} tool calls reached."
                rprint(f"[blue]\[tool response -->] {tool_output}[/blue]\n")
                retry_call = False

            if response.content != None:
                tool_result = [{
                    "role": "assistant",
                    "content": response.content
                }, {"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": tool_call.type, "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}}]}, {"role": "tool", "tool_call_id": tool_call.id, "content": tool_output}]
            else:
                tool_result = [{"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": tool_call.type, "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}}]}, {"role": "tool", "tool_call_id": tool_call.id, "content": tool_output}]
            return tool_name, tool_result, retry_call
            
        elif response.content != None:
            # If no function call, it's a normal text response
            assistant_content = response.content
            # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
            rprint(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            rprint(f"[green]'{assistant_content}'[/green]\n")
            return None, {"role": "assistant", "content": assistant_content}, False
        else:
            return None, None, False

