from rich import print as rprint
import json
from collections import defaultdict
import copy
import re
from agents.api_conf import APIConfiguration
from agents.agent import Agent
import litellm

def get_response_json(response):
    try:
        out = json.loads(response)
    except json.JSONDecodeError:
        # Extract JSON-like content from the text using regex
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_match = json_match.group()
        # json_match = json_match.replace('\'', '"')
        json_str = re.sub(r"(?<!\\)'", '"', json_match)
        if json_match:
            out = json.loads(json_match)
        else:
            raise
    return out

class Executor:
    def __init__(self, agent, openai_tools, terminate_function, target_function, api_conf:APIConfiguration):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        self.target_function = target_function
        self.max_retries = 3
        self.max_interaction_retries = 12

        self.number_of_consecutive_messages = 0
        self.curr_interaction_retries = 0

        self.api_conf = api_conf
        self.client = Agent(self.api_conf)

    def step(self, observation: str):
        self.agent.was_last_message = False
        # Call the agent's tools or send messages
        self.agent.update_history({"role": "user", "content": observation})
        rprint(f"[bold magenta]\[`user` -> `{self.agent.name.lower()}`][/bold magenta]")
        rprint(f"[green]'{observation}'[/green]\n")

        self.filtered_tools = copy.deepcopy(self.openai_tools)
        response = self._call_llm()
        self.function_call_count = defaultdict(int)
        function_name, processed_response, retry_call = self._process_llm_response(response)
        # if function_name != None:
            # self.function_call_count[function_name] += 1

        # num_retries = 0
        while retry_call:
            self.curr_interaction_retries += 1
            if self.curr_interaction_retries > self.max_interaction_retries:
                print(f"[bold red]Maximum number of interaction retries reached in episode. Exiting...[/bold red]\n")
                return False

            # print("number_of_consecutive_messages", self.number_of_consecutive_messages)
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
            # num_retries += 1
            # if num_retries > self.max_retries:
            #     break

        if processed_response != None:
            self.agent.update_history(processed_response)

        if function_name == self.terminate_function:
            return True
        return False
    
    def _call_llm(self):
        # start_time = time.time()
        retries = 0
        while True:
            retries +=1
            if retries > 5:
                print(f"[bold red]Maximum number of retries reached when calling LLM. Exiting ...[/bold red]\n")
                return None
            try:
                response = self.client(query=self.agent.history, tools=self.filtered_tools, tool_choice="auto")
                break
            except litellm.RateLimitError:
                print(f"[bold red]Rate limit error when calling LLM. Retrying...[/bold red]\n")
                continue
            except Exception as e:
                print(f"[bold red]Error calling LLM: {e}[/bold red]\n")
                continue

            # try:
            #     out = response.choices[0].message
            #     break
            # except Exception as e:
            #     print(f"[bold red]Error processing LLM response: {e}[/bold red]\n")
            #     continue

        # self.total_prompt_tokens += response.usage.prompt_tokens
        # self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message
    
    def _process_llm_response(self, response):
        tool_call = response.tool_calls
        # Check if the assistant returned a function call
        if tool_call != None:
            self.agent.was_last_message = False
            retry_call = True
            if len(response.tool_calls) > 1:
                print(f"[bold yellow]Received multiple tool calls: {response.tool_calls}. Will only process the first item in the list.[/bold yellow]\n")
                # return None, None, True
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"[bold red]\Error decoding arguments JSON: {e}[/bold red]\n")
                arguments = {}
            if tool_name not in self.agent.tools:
                raise ValueError(f"Function {tool_name} not found in tools")
            
            if tool_name == self.terminate_function:
                print(f"[bold red]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/bold red]\n")
                tool_output = None
                retry_call = False
            else:
                tool = self.agent.tools[tool_name]
                print(f"[blue]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/blue]")
                tool_output = tool(**arguments)
                self.function_call_count[tool_name] += 1
                if self.function_call_count[tool_name] > self.max_retries:
                    tool_output = f'{{"status": "429", "message": "Maximum number of retries reached for the tool {tool_name}. Please try again later."}}'
                    self.filtered_tools = [tool for tool in self.filtered_tools if
                                           tool['function']['name'] != tool_name]  # remove the tool from the list
                # if tool_name == self.target_function:
                #     self.target_function_count += 1
                # if self.target_function_count > self.max_target_tries:
                #     tool_output = f"{{'status': '429', 'message': 'Maximum number of retries reached for the tool {tool_name}. Please try again later.'}}"
                print(f"[blue]\[tool response -->] {tool_output}[/blue]\n")
                retry_call = True

            # Still processing tool call
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

            try:
                assistant_content_json = get_response_json(assistant_content)
            except Exception as e:
                print(f"[bold red]Error decoding assistant message JSON: {e}.\nRetrying...[/bold red]\n")
                print(f"[bold red]Assistant message:\n{assistant_content}[/bold red]\n")
                return None, None, True
            try:
                assert set(assistant_content_json.keys()) == {"thoughts", "message", "interaction_status"}
            except Exception as e:
                print(
                    f"[bold red]Error: Assistant message JSON does not contain the expected keys. Retrying...[/bold red]\n")
                print(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, None, True
            try:
                assert assistant_content_json['interaction_status'] in ["continue", "terminate"]
            except Exception as e:
                print(
                    f"[bold red]Error: Assistant message JSON does not contain the expected interaction status. Retrying...[/bold red]\n")
                print(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, {"role": "user",
                              "content": "Backend System: Invalid interaction status received; MUST be set to either \"continue\" or \"terminate\". Try again."}, True

            # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
            print(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            print(f"[green]'{json.dumps(assistant_content_json, indent=2)}'[/green]\n")

            if assistant_content_json['interaction_status'] == "terminate":
                self.agent.was_last_message = False
                return None, {"role": "assistant", "content": assistant_content}, True

            # Task is not terminated, so we can continue
            if not self.agent.was_last_message:
                self.agent.was_last_message = True
                message_received_message = 'Backend System: Message received.'

                # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
                print(f"[magenta]\[`user` -> `{self.agent.name.lower()}`][/magenta]")
                print(f"[green]'{message_received_message}'[/green]\n")

                return None, [{"role": "assistant", "content": assistant_content}, {"role": "user", "content": message_received_message}], True
            else:
                self.agent.was_last_message = False

            return None, {"role": "assistant", "content": assistant_content}, False
        else:
            return None, None, False

