"""
Lightweight Agent implementation for dynamic tool management and execution.
"""

import json
import inspect
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass
import random


SYSTEM_PROMPT_TEMPLATE = """
You are an AI agent named {role}.
Your role is:
{role_policy}

You have the following duties:
{duties}

You have the following constraints:
{constraints} 

You are operating in the following scenario:
{scenario_description}

You have the following tools:
{tools}

When executing tasks:
1. Always operate within your defined constraints
2. Follow the established role policy
3. Fulfill your duties effectively
4. Use available tools appropriately to achieve your goals
5. Validate inputs and handle errors gracefully
6. Provide clear feedback about your actions 

For each tool you use:
- Verify it aligns with your role and policies
- Ensure it respects all constraints
- Confirm you have all required parameters
- Execute within your defined scope 

If a conversation has reached an outcome it can be ended by not responding to the agent.
Otherwise respond with at least one tool call or message.

You can also end the conversation by not responding to the agent.
"""

@dataclass
class Tool:
    """
    Represents a tool that can be used by the agent.
    
    Attributes:
        name: The name of the tool
        function: The actual function to be called
        description: A description of what the tool does
        parameters: Dictionary containing parameter information
    """
    name: str
    function: Callable
    description: str
    parameters: Dict[str, Any]

class Agent:
    """
    A lightweight agent that can dynamically manage and execute tools.
    """
    
    def __init__(self, scenario):
        self.scenario = scenario
        self.name = scenario['agent_info']['role']
        self.system_prompt = self.create_system_prompt(scenario)
        self.tools = self.add_tools(scenario)
        self.history = [{"role": "system", "content": self.system_prompt}]

    def create_system_prompt(self, scenario):
        tools_str = "\n".join([f"{tool_info['function']}: {tool_info['description']}" for tool_info in scenario['tools'].values()])
        return SYSTEM_PROMPT_TEMPLATE.format(
            role=scenario['agent_info']['role'],
            role_policy=scenario['agent_info']['role_policy'],
            duties=scenario['agent_info']['duties'],
            constraints=scenario['agent_info']['constraints'],
            scenario_description=scenario['metadata']['description'],
            tools=tools_str
        )

    def add_tools(self, scenario):
        tools = {}
        # Construct tools from scenario description
        for tool_info in scenario['tools'].values():
            tools[tool_info['function']] = self._construct_tool(
                name=tool_info['function'],
                description=tool_info['description'],
                parameters=tool_info['parameters'],
                output=tool_info['output']
            )
        return tools
    
    def _construct_tool(self, name: str, description: str, parameters: Optional[Dict[str, Any]], output: Union[List[str], str]) -> Callable:
        """
        Construct a tool based on the tool information provided in the scenario.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            parameters: Dictionary of parameters for the tool
            output: Expected output type and values
        
        Returns:
            A callable function that represents the tool
        """
        def tool_function(**kwargs):
            # Validate required parameters
            for param_name, param_details in parameters.items():
                if param_details.get("required", False) and param_name not in kwargs:
                    print(f"Error: Required parameter '{param_name}' not provided.")
                    return None

                if param_name in kwargs:
                    param_type = param_details.get("type")
                    # Convert string type to actual type
                    if param_type == "string":
                        param_type = str
                    elif param_type == "float":
                        param_type = float
                    elif param_type == "integer":
                        param_type = int
                    else:
                        print(f"Error: Unsupported parameter type '{param_type}'.")
                        return None
                    
                    if not isinstance(kwargs[param_name], param_type):
                        print(f"Error: Parameter '{param_name}' must be a {param_type.__name__}.")
                        return None

            # Generate output based on the output type
            if output['type'] == "list":
                return random.choice(output['values'])

            if output['type'] == "null":
                return None

        # Set function name and docstring
        tool_function.__name__ = name
        tool_function.__doc__ = description
        return tool_function

    def update_history(self, messages: Union[str, List[Dict[str, str]]]) -> None:
        """
        Update the history of the agent.
        
        Args:
            observation: The observation of the agent
        """
        if isinstance(messages, list):
            self.history.extend(messages)
        else:   
            self.history.append(messages)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())