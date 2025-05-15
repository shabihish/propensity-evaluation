from rich import print as rprint
# from llm_client import LiteLlmClient
import litellm
from agents.agent import Agent
from agents.api_conf import APIConfiguration
import time
import json
import os
from collections import defaultdict
import copy
import re


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
    def __init__(self, agent, openai_tools, terminate_function, target_function, model_name):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        self.target_function = target_function
        self.model_name = model_name
        self.max_retries = 3
        self.max_interactin_retries = 8

        self.number_of_consecutive_messages = 0
        self.curr_interaction_retries = 0

        # self.client = LiteLlmClient(model_name)

        api_conf = APIConfiguration(
            model_name=os.path.basename(model_name),
            model_provider=os.path.dirname(model_name),
            api_base=None,
            use_cache=False,

        )
        self.client = Agent(api_conf)

        # self.max_target_tries = 4
        # Initialize cost tracking
        # self.total_prompt_tokens = 0
        # self.total_completion_tokens = 0

    # def get_cost_info(self):
    #     return {
    #         'prompt_tokens': self.total_prompt_tokens,
    #         'completion_tokens': self.total_completion_tokens,
    #     }

    def step(self, observation: str):
        self.number_of_consecutive_messages = 0
        self.curr_interaction_retries = 0

        # Call the agent's tools or send messages
        self.agent.update_history({"role": "user", "content": observation})
        rprint(f"[bold magenta]\[`user` -> `{self.agent.name.lower()}`][/bold magenta]")
        rprint(f"[green]'{observation}'[/green]\n")

        self.filtered_tools = copy.deepcopy(self.openai_tools)
        response = self._call_llm()

        self.function_call_count = defaultdict(int)
        function_name, processed_response, retry_call = self._process_llm_response(response)
        if function_name != None:
            self.function_call_count[function_name] += 1

        # num_retries = 0
        while retry_call:
            self.curr_interaction_retries += 1
            if self.curr_interaction_retries > self.max_interactin_retries:
                rprint(f"[bold red]Maximum number of interaction retries reached in episode. Exiting...[/bold red]\n")
                return False

            # print("number_of_consecutive_messages", self.number_of_consecutive_messages)
            if processed_response == None:
#                 print("Re-calling LLM (none response)")
                response = self._call_llm()
                function_name, processed_response, retry_call = self._process_llm_response(response)
            elif function_name == self.terminate_function:
#                 print("terminate_function detected.")
                self.agent.update_history(processed_response)
                return True
            else:
                self.agent.update_history(processed_response)
#                 print("Re-calling LLM (retry)")
                response = self._call_llm()
                # print(f"Got retry response: {response}")
#                 print(f"Got retry response")
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
                rprint(f"[bold red]Maximum number of retries reached when calling LLM. Exiting ...[/bold red]\n")
                return None
            # print("Calling LLM in while loop...")
            try:
                if isinstance(self.client, Agent):
                    response = self.client(query=self.agent.history, tools=self.filtered_tools, tool_choice="auto")
                else:
                    response = self.client.get_completion(messages=self.agent.history, tools=self.filtered_tools,
                                                          tool_choice="auto")
            except litellm.RateLimitError:
                rprint(f"[bold red]Rate limit error when calling LLM. Retrying...[/bold red]\n")
                continue
            except Exception as e:
                rprint(f"[bold red]Error calling LLM: {e}[/bold red]\n")
                continue

            try:
                out = response.choices[0].message
                break
            except Exception as e:
                rprint(f"[bold red]Error processing LLM response: {e}[/bold red]\n")
                continue

        # self.total_prompt_tokens += response.usage.prompt_tokens
        # self.total_completion_tokens += response.usage.completion_tokens
        # print("Break out of while loop, got response")
        return out


    def _process_llm_response(self, response):
        if response is None:
            return None, None, False

        tool_call = response.tool_calls
        # Check if the assistant returned a function call
        if tool_call != None:
            self.number_of_consecutive_messages = 0
            # self.agent.was_last_message = False
            retry_call = True
            if len(response.tool_calls) > 1:
                rprint(f"[bold yellow]Received multiple tool calls: {response.tool_calls}[/bold yellow]\n")
                # return None, None, True
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                rprint(f"[bold red]\Error decoding arguments JSON: {e}[/bold red]\n")
                arguments = {}
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
                self.function_call_count[tool_name] += 1
                if self.function_call_count[tool_name] >= self.max_retries:
                    tool_output = f'{{"status": "429", "message": "Maximum number of retries reached for the tool {tool_name}. Please try again later."}}'
                    self.filtered_tools = [tool for tool in self.filtered_tools if
                                           tool['function']['name'] != tool_name]  # remove the tool from the list
                # if tool_name == self.target_function:
                #     self.target_function_count += 1
                # if self.target_function_count > self.max_target_tries:
                #     tool_output = f"{{'status': '429', 'message': 'Maximum number of retries reached for the tool {tool_name}. Please try again later.'}}"
                rprint(f"[blue]\[tool response -->] {tool_output}[/blue]\n")
                retry_call = True

            if response.content != None:
                tool_result = [{
                    "role": "assistant",
                    "content": response.content
                }, {"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": tool_call.type,
                                                         "function": {"name": tool_call.function.name,
                                                                      "arguments": tool_call.function.arguments}}]},
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_output, indent=2)}]
            else:
                tool_result = [{"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": tool_call.type,
                                                                     "function": {"name": tool_call.function.name,
                                                                                  "arguments": tool_call.function.arguments}}]},
                               {"role": "tool", "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_output, indent=2)}]
            return tool_name, tool_result, retry_call

        elif response.content != None:
            # If no function call, it's a normal text response
            assistant_content = response.content
            try:
                assistant_content_json = get_response_json(assistant_content)
            except Exception as e:
                rprint(f"[bold red]Error decoding assistant message JSON: {e}.\nRetrying...[/bold red]\n")
                rprint(f"[bold red]Assistant message:\n{assistant_content}[/bold red]\n")
                return None, None, True

            try:
                assert set(assistant_content_json.keys()) == {"thoughts", "message", "interaction_status"}
            except Exception as e:
                rprint(
                    f"[bold red]Error: Assistant message JSON does not contain the expected keys. Retrying...[/bold red]\n")
                rprint(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, None, True

            try:
                assert assistant_content_json['interaction_status'] in ["continue", "terminate"]
            except Exception as e:
                rprint(
                    f"[bold red]Error: Assistant message JSON does not contain the expected interaction status. Retrying...[/bold red]\n")
                rprint(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, {"role": "user",
                              "content": "Backend System: Invalid interaction status received; MUST be set to either \"continue\" or \"terminate\". Try again."}, True

            # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
            rprint(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            rprint(f"[green]'{json.dumps(assistant_content_json, indent=2)}'[/green]\n")

            if assistant_content_json['interaction_status'] == "terminate":
                # self.agent.was_last_message = False
                self.number_of_consecutive_messages = 0
                return None, {"role": "assistant", "content": assistant_content}, False

            # Task is not terminated, so we can continue
            if self.number_of_consecutive_messages <= 2:
                self.number_of_consecutive_messages += 1
                message_received_message = 'Backend System: Message received.'

                # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
                rprint(f"[magenta]\[`user` -> `{self.agent.name.lower()}`][/magenta]")
                rprint(f"[green]'{message_received_message}'[/green]\n")

                return None, [{"role": "assistant", "content": assistant_content},
                              {"role": "user", "content": message_received_message}], True
            else:
                self.number_of_consecutive_messages = 0
                # self.agent.was_last_message = False
                return None, {"role": "assistant", "content": assistant_content}, False
        else:
            return None, None, False
