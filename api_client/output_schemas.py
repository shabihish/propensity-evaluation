from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Type
from pathlib import Path
import os

# Mapping JSON schema types to Python types
type_mapping = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict
}

def generate_pydantic_model(name: str, json_obj: Dict[str, Any]) -> Type[BaseModel]:
    def parse_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        fields = {}
        for key, value in properties.items():
            if value["type"] == "object":
                fields[key] = (generate_pydantic_model(key.capitalize(), value), ...)
            elif value["type"] == "array":
                item_type = value["items"]["type"]
                if item_type == "object":
                    fields[key] = (List[generate_pydantic_model(key.capitalize(), value["items"])], ...)
                else:
                    fields[key] = (List[type_mapping[item_type]], ...)
            else:
                fields[key] = (type_mapping[value["type"]], ...)
        return fields

    properties = json_obj.get("properties", {})
    required = json_obj.get("required", [])
    fields = parse_properties(properties)

    return create_model(name, **fields)


from datamodel_code_generator import InputFileType, generate


def generate_pydantic_model_from_json(json_schema: str) -> str:
    assert isinstance(json_schema, str), "json_schema must be a string"

    # output_buffer = StringIO()

    # with redirect_stdout(output_buffer):
        # Generate the Pydantic model (this will print the code to stdout)
    generate(
        input_=json_schema,
        input_file_type=InputFileType.JsonSchema,
        output=Path('.tmp/pydantic_model.py')
    )

    output_file_path = '.tmp/pydantic_model.py'
    with open(output_file_path, 'r+') as f:
        f.flush()
        os.fsync(f.fileno())

    with open('.tmp/pydantic_model.py', 'r') as f:
        output_code = f.read()
    # output_code = str(output_buffer.getvalue())
    print(output_code)  # Print the generated code for debugging
    has_executed = False
    while not has_executed:
        try:
            exec(output_code, globals())
            has_executed = True
        except Exception as e:
            print(f"Error executing generated code for Pydantic classes. Retrying...")
    return Model
