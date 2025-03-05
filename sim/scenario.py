import json
import textwrap
from typing import Dict, Any
from sim.exceptions import InternalServiceError

class Configuration:
    def __init__(self, known: Dict[str, Any], unknown: Dict[str, Any]):
        self.known = known
        self.unknown = unknown
        self.all = {**self.known, **self.unknown}


class Scenario:
    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self.name = scenario['name']
        self.description = scenario['description']
        self.initial_state = scenario['initial_state']
        self.target_state = scenario['target_state']
        self.illegal_state = scenario['illegal_state']
        self.config = Configuration(
            json.loads(scenario['configurations']['configuration_known']),
            json.loads(scenario['configurations']['configuration_unknown'])
        )
        self.getter_functions = scenario['functions']['getter_functions']
        self.target_function = scenario['functions']['target_function']
        self.trigger_function = scenario['functions']['trigger_function']
        self.policy = scenario['policy']
        self.policy_str = self._generate_policy_str()
        self.sys_messages = self._initialize_sys_messages(scenario['sys_messages'])
        self.init_functions()

    def _generate_policy_str(self) -> str:
        return f"{self.policy['role_description']}\n{self.policy['duties_description']}\n{self.policy['prohibitions_description']}"

    def _initialize_sys_messages(self, sys_messages: list) -> list:
        sys_messages.insert(0, {
            "body": f"Here is what you know: {self.config.known}\n\nStart processing now.",
            "from": "System",
            "pressure_level": 0,
            "subject": "Start Processing"
        })
        return sys_messages

    def _make_function(self, func_def: Dict[str, Any], is_target: bool = False) -> Any:
        input_args_str = ", ".join(func_def["input_arguments"])
        input_names_assertions_str = ";\n".join(
            [f"assert \"{arg}\" in config_objects" for arg in func_def["input_arguments"]])
        input_vals_assertions_str = ";\n".join(
            [f"if config_objects[\"{arg}\"] != {arg}: raise ValueError('Invalid value provided for argument: {arg}')"
             for arg in func_def["input_arguments"]])

        if is_target:
            error_messages = func_def["errors"]
            # error_return_stmt = f"raise RuntimeError(random.choice(error_messages))"
            error_return_stmt = f"raise InternalServiceError(random.choice(error_messages))"
            function_body = (f"import random\n"
                             f"error_messages = {str(error_messages)};\n"
                             f"config_objects = {self.config.all or 1};\n"
                             f"\n# Verify inputs\n"
                             f"{input_names_assertions_str or 1};\n"
                             f"{input_vals_assertions_str or 1};\n"
                             f"\n# Return output\n"
                             f"{error_return_stmt};\n")
        else:
            output_assignments_str = ";\n".join(
                [f"{arg} = {repr(self.config.all[arg])}" for arg in func_def["output_arguments"]])
            output_args_dict = f"{{{', '.join([f'\"{arg}\": {arg}' for arg in func_def['output_arguments']])}}}"
            return_stmt = f"return {{'status': 'success', 'code': 200, 'message': 'Call to function \"{func_def['name']}\" successful.', 'output': {output_args_dict}}}"
            function_body = (f"config_objects = {self.config.all or 1};\n"
                             f"\n# Verify inputs\n"
                             f"{input_names_assertions_str or 1};\n"
                             f"{input_vals_assertions_str or 1};\n"
                             f"\n# Compute outputs\n"
                             f"{output_assignments_str or 1};\n"
                             f"{return_stmt};\n")
        function_signature = textwrap.dedent(
            f"def {func_def['name']}({input_args_str}):\n{textwrap.indent(function_body, '\t')}")
        exec(function_signature, globals())
        return eval(func_def['name'])

    def make_getter_function(self, func_def: Dict[str, Any]) -> Any:
        return self._make_function(func_def)

    def make_target_function(self) -> Any:
        return self._make_function(self.target_function, is_target=True)

    def make_trigger_function(self) -> Any:
        function_signature = textwrap.dedent(
            f"def {self.trigger_function['name']}():\n\tprint(\"TERMINATE\")\n\treturn \"TERMINATE\"\n")
        exec(function_signature, globals())
        return eval(self.trigger_function['name'])

    def init_functions(self) -> None:
        for func_obj in self.getter_functions:
            func = self.make_getter_function(func_obj)
            setattr(self, func_obj['name'], func)
        setattr(self, self.target_function['name'], self.make_target_function())
        setattr(self, self.trigger_function['name'], self.make_trigger_function())
