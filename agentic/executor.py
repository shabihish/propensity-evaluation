# import json
# import logging
# import re
# import time
# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple
#
# import litellm
# from rich.logging import RichHandler
#
# from agents.agent import Agent
# from agents.api_conf import APIConfiguration
#
# def setup_logger(log_file='app.log'):
#     # Use rich for pretty, colored logging in the console.
#     logger = logging.getLogger()
#     logging.getLogger().setLevel(logging.DEBUG)
#     for logger_name in logging.root.manager.loggerDict:
#         print(f'Muting logs from {logger_name}')
#         logging.getLogger(logger_name).setLevel(logging.CRITICAL)
#
#     logging.basicConfig(
#         level="INFO",
#         format="%(message)s",
#         datefmt="[%X]",
#         handlers=[RichHandler(rich_tracebacks=True)],
#     )
#
# setup_logger()
# log = logging.getLogger("rich")
#
# # --- Custom Exceptions for Clearer Error Handling ---
# class LLMCallError(Exception):
#     """Raised when the LLM fails to provide a valid response after all retries."""
#     pass
#
#
# class MalformedResponseError(ValueError):
#     """Raised when the LLM's response content is not in the expected format."""
#     pass
#
#
# # --- Configuration Dataclass to Remove Magic Numbers ---
# @dataclass
# class ExecutorConfig:
#     """Configuration for the Executor's behavior."""
#     max_llm_retries: int = 5
#     max_interaction_retries: int = 12
#     max_tool_retries: int = 3
#     max_consecutive_messages: int = 2
#
#
# def get_response_json(response_text: str) -> Dict[str, Any]:
#     """
#     Parses a string to JSON, with a regex fallback for common LLM errors.
#
#     Args:
#         response_text: The string content from the LLM.
#
#     Returns:
#         A dictionary parsed from the text.
#
#     Raises:
#         MalformedResponseError: If the text cannot be parsed as JSON.
#     """
#     try:
#         return json.loads(response_text)
#     except json.JSONDecodeError:
#         # Fallback: Extract JSON-like content using regex. This is brittle but
#         # can rescue responses where the LLM adds text outside the JSON object.
#         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#         if not json_match:
#             raise MalformedResponseError("No JSON object found in the response text.")
#
#         json_str = json_match.group(0)
#         try:
#             return json.loads(json_str)
#         except json.JSONDecodeError as e:
#             raise MalformedResponseError(f"Failed to decode extracted JSON: {e}")
#
#
# class Executor:
#     """
#     Orchestrates the interaction between an LLM agent and a set of tools
#     to accomplish a task based on user observations.
#     """
#
#     def __init__(self, agent: Agent, openai_tools: List[Dict], terminate_function: str,
#                  target_function: str, api_conf: APIConfiguration, config: Optional[ExecutorConfig] = None):
#         self.agent = agent
#         self.openai_tools = openai_tools
#         self.terminate_function = terminate_function
#         self.target_function = target_function  # Note: This was unused in the original code
#         self.config = config or ExecutorConfig()
#
#         self.api_conf = api_conf
#         self.client = Agent(self.api_conf, temperature=0)
#
#         # State variables, reset for each step
#         self.number_of_consecutive_messages = 0
#         self.function_call_count = defaultdict(int)
#         self.filtered_tools = []
#
#     def step(self, observation: str) -> bool:
#         """
#         Executes a single turn of the agent-environment interaction.
#
#         Args:
#             observation: The new information or prompt from the user/environment.
#
#         Returns:
#             True if the agent decided to terminate the task, False otherwise.
#
#         Raises:
#             LLMCallError: If the underlying LLM call fails definitively.
#         """
#         # Reset step-specific state
#         self.number_of_consecutive_messages = 0
#         self.function_call_count.clear()
#         self.filtered_tools = list(self.openai_tools)
#
#         self.agent.update_history({"role": "user", "content": observation})
#         log.info(f"[`user` -> `{self.agent.name.lower()}`] {observation}\n")
#
#         for i in range(self.config.max_interaction_retries):
#             # This call will now raise LLMCallError on persistent failure
#             llm_response = self._call_llm()
#
#             history_update, continue_interaction, func_name = self._process_llm_response(llm_response)
#
#             if history_update:
#                 self.agent.update_history(history_update)
#
#             if func_name == self.terminate_function or not continue_interaction:
#                 # Task has ended, either by termination or by reaching a stable state.
#                 return func_name == self.terminate_function
#
#         log.warning(
#             f"Maximum number of interaction retries ({self.config.max_interaction_retries}) reached. "
#             "Ending turn.\n"
#         )
#         return False
#
#     def _call_llm(self):
#         """
#         Calls the language model with retry logic.
#
#         Returns:
#             The message object from the LLM response.
#
#         Raises:
#             LLMCallError: If the call fails after all retries.
#         """
#         for attempt in range(self.config.max_llm_retries):
#             try:
#                 response = self.client(query=self.agent.history,
#                                        tools=self.filtered_tools,
#                                        tool_choice="auto"
#                                        )
#                 # Ensure the response is valid before returning
#                 if response.choices and response.choices[0].message:
#                     return response.choices[0].message
#                 else:
#                     log.warning(
#                         f"LLM returned an invalid or empty response object. Retrying... (Attempt {attempt + 1})")
#
#             except litellm.RateLimitError:
#                 log.warning(f"Rate limit error. Retrying... (Attempt {attempt + 1})\n")
#                 time.sleep(2 ** attempt)  # Exponential backoff
#             except Exception as e:
#                 log.error(f"Error calling LLM: {e}. Retrying... (Attempt {attempt + 1})\n")
#                 time.sleep(1)
#
#         raise LLMCallError(f"Failed to get a valid response from LLM after {self.config.max_llm_retries} retries.")
#
#     def _process_llm_response(self, response) -> Tuple[
#         List[Dict], bool, Optional[str]]:
#         """
#         Processes the LLM's response, delegating to tool or content handlers.
#
#         Returns:
#             A tuple containing:
#             - list: The messages to be added to the agent's history.
#             - bool: A flag indicating whether to continue the interaction loop.
#             - str | None: The name of the function called, if any.
#         """
#         if response.tool_calls:
#             if len(response.tool_calls) > 1:
#                 log.warning(
#                     f"Received multiple tool calls, processing only the first: {response.tool_calls}\n")
#             return self._handle_tool_call(response.tool_calls[0], response.content)
#
#         elif response.content:
#             return self._handle_content_response(response.content)
#
#         else:
#             # LLM returned a message with neither content nor tool calls.
#             log.warning("LLM response was empty. Requesting a retry.")
#             return [], True, None
#
#     def _handle_tool_call(self, tool_call, response_content: Optional[str]) -> Tuple[List[Dict], bool, Optional[str]]:
#         """Handles the logic when the LLM requests a tool call."""
#         self.number_of_consecutive_messages = 0
#         tool_name = tool_call.function.name
#
#         try:
#             arguments = json.loads(tool_call.function.arguments)
#         except json.JSONDecodeError as e:
#             log.error(f"Error decoding tool arguments JSON: {e}\n")
#             arguments = {}
#
#         if tool_name == self.terminate_function:
#             log.info(f"[{self.agent.name.lower()}: tool call -->] {tool_name}\n")
#             # Final message to history, but the loop should not continue.
#             history_update = self._format_tool_history(tool_call, None, response_content)
#             return history_update, False, tool_name
#
#         log.info(f"[{self.agent.name.lower()}: tool call -->] {tool_name}")
#         self.function_call_count[tool_name] += 1
#
#         # Check if tool is available and hasn't exceeded retries
#         if tool_name not in self.agent.tools:
#             tool_output = f'{{"status": "404", "message": "Function {tool_name} not found."}}'
#         elif self.function_call_count[tool_name] > self.config.max_tool_retries:
#             tool_output = f'{{"status": "429", "message": "Max retries for tool {tool_name} reached."}}'
#             # Remove the tool from future consideration in this step
#             self.filtered_tools = [t for t in self.filtered_tools if t['function']['name'] != tool_name]
#         else:
#             tool = self.agent.tools[tool_name]
#             tool_output = tool(**arguments)
#
#         log.info(f"[tool response -->] {json.dumps(tool_output)}\n")
#
#         history_update = self._format_tool_history(tool_call, tool_output, response_content)
#         return history_update, True, tool_name
#
#     def _handle_content_response(self, content: str) -> Tuple[List[Dict], bool, Optional[str]]:
#         """Handles the logic when the LLM returns a text-based message."""
#         try:
#             content_json = get_response_json(content)
#
#             # Validate structure
#             if not isinstance(content_json, dict) or {"thoughts", "message",
#                                                       "interaction_status"} != content_json.keys():
#                 raise MalformedResponseError("JSON does not contain the expected keys.")
#             if content_json['interaction_status'] not in ["continue", "terminate"]:
#                 raise MalformedResponseError("Invalid 'interaction_status' value.")
#
#         except MalformedResponseError as e:
#             log.error(f"Error in assistant message format: {e}. Requesting retry.\n")
#             # Provide feedback to the model to correct itself
#             feedback = f"Backend System: Invalid response format. Error: {e}. Ensure your response is a single JSON object with keys 'thoughts', 'message', and 'interaction_status' set to 'continue' or 'terminate'."
#             return [{"role": "user", "content": feedback}], True, None
#
#         log.info(f"[`{self.agent.name.lower()}` -> `user`] {json.dumps(content_json, indent=2)}\n")
#
#         assistant_message = {"role": "assistant", "content": content}
#
#         if content_json['interaction_status'] == "terminate":
#             self.number_of_consecutive_messages = 0
#             return [assistant_message], False, None
#
#         # If the agent is just talking, limit how many times it can do so without taking action.
#         if self.number_of_consecutive_messages < self.config.max_consecutive_messages:
#             self.number_of_consecutive_messages += 1
#             user_feedback = {"role": "user", "content": "Backend System: Message received."}
#             log.info(f"[`user` -> `{self.agent.name.lower()}`] {user_feedback['content']}\n")
#             return [assistant_message, user_feedback], True, None
#         else:
#             # Reached max consecutive messages, end the turn without termination.
#             log.warning(
#                 f"Max consecutive messages ({self.config.max_consecutive_messages}) reached. Ending turn.\n")
#             self.number_of_consecutive_messages = 0
#             return [assistant_message], False, None
#
#     @staticmethod
#     def _format_tool_history(tool_call, tool_output: Any, content: Optional[str]) -> List[Dict]:
#         """Formats the tool call and result into the OpenAI message format."""
#         assistant_message = {
#             "role": "assistant",
#             "content": content,
#             "tool_calls": [{
#                 "id": tool_call.id,
#                 "type": tool_call.type,
#                 "function": {
#                     "name": tool_call.function.name,
#                     "arguments": tool_call.function.arguments,
#                 },
#             }],
#         }
#         if content is None:
#             del assistant_message['content']
#
#         tool_message = {
#             "role": "tool",
#             "tool_call_id": tool_call.id,
#             "content": json.dumps(tool_output, indent=2),
#         }
#         return [assistant_message, tool_message]






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
# from llm_client import LiteLlmClient

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
        self.number_of_consecutive_messages = 0
        self.curr_interaction_retries = 0

        # Call the agent's tools or send messages
        self.agent.update_history({"role": "user", "content": observation})
        print(f"[bold magenta]\[`user` -> `{self.agent.name.lower()}`][/bold magenta]")
        print(f"[green]'{observation}'[/green]\n")

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
                print(f"[bold red]Maximum number of retries reached when calling LLM. Exiting ...[/bold red]\n")
                raise Exception("Maximum number of retries reached when calling LLM.")
                # return None
            # print("Calling LLM in while loop...")
            # import pdb
            # pdb.set_trace()
            try:
                if isinstance(self.client, Agent):
                    response = self.client(query=self.agent.history, tools=self.filtered_tools, tool_choice="auto")
                else:
                    response = self.client.get_completion(messages=self.agent.history, tools=self.filtered_tools,
                                                          tool_choice="auto")
            except litellm.RateLimitError:
                print(f"[bold red]Rate limit error when calling LLM. Retrying...[/bold red]\n")
                continue
            except Exception as e:
                print(f"[bold red]Error calling LLM: {e}[/bold red]\n")
                continue

            try:
                out = response.choices[0].message
                break
            except Exception as e:
                print(f"[bold red]Error processing LLM response: {e}[/bold red]\n")
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
                print(f"[bold yellow]Received multiple tool calls: {response.tool_calls}[/bold yellow]\n")
                # return None, None, True
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"[bold red]\Error decoding arguments JSON: {e}[/bold red]\n")
                arguments = {}

            if tool_name == self.terminate_function:
                print(f"[bold red]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/bold red]\n")
                tool_output = None
                retry_call = False
            else:
                print(f"[blue]\[{self.agent.name.lower()}: tool call -->] {tool_name}[/blue]")
                if tool_name not in self.agent.tools:
                    tool_output = f'{{"status": "404", "message": "Function {tool_name} not found in tools.  Please try again later."}}'
                else:
                    tool = self.agent.tools[tool_name]
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
                print(f"[bold red]Error decoding assistant message JSON: {e}.\nRetrying...[/bold red]\n")
                # print(f"[bold red]Assistant message:\n{assistant_content}[/bold red]\n")
                return None, None, True

            try:
                assert set(assistant_content_json.keys()) == {"thoughts", "message", "interaction_status"}
            except Exception as e:
                print(
                    f"[bold red]Error: Assistant message JSON does not contain the expected keys. Retrying...[/bold red]\n")
                # print(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, None, True

            try:
                assert assistant_content_json['interaction_status'] in ["continue", "terminate"]
            except Exception as e:
                print(
                    f"[bold red]Error: Assistant message JSON does not contain the expected interaction status. Retrying...[/bold red]\n")
                # print(f"[bold red]Assistant message JSON:\n{assistant_content_json}[/bold red]\n")
                return None, {"role": "user",
                              "content": "Backend System: Invalid interaction status received; MUST be set to either \"continue\" or \"terminate\". Try again."}, True

            # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
            print(f"[magenta]\[`{self.agent.name.lower()}` -> `user`][/magenta]")
            print(f"[green]'{json.dumps(assistant_content_json, indent=2)}'[/green]\n")

            if assistant_content_json['interaction_status'] == "terminate":
                # self.agent.was_last_message = False
                self.number_of_consecutive_messages = 0
                return None, {"role": "assistant", "content": assistant_content}, False

            # Task is not terminated, so we can continue
            if self.number_of_consecutive_messages <= 2:
                self.number_of_consecutive_messages += 1
                message_received_message = 'Backend System: Message received.'

                # tool_name, arguments = self._parse_content_for_tool_call(assistant_content)
                print(f"[magenta]\[`user` -> `{self.agent.name.lower()}`][/magenta]")
                print(f"[green]'{message_received_message}'[/green]\n")

                return None, [{"role": "assistant", "content": assistant_content},
                              {"role": "user", "content": message_received_message}], True
            else:
                self.number_of_consecutive_messages = 0
                # self.agent.was_last_message = False
                return None, {"role": "assistant", "content": assistant_content}, False
        else:
            return None, None, False
