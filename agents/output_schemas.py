# from pydantic import BaseModel
#
#
# class RolesGenerationAgent:
#
#     pass
#
#
# class RolesRevisionAgent:
#     pass
#
#
# class ScenariosGenerationAgent:
#     pass
#
#
# class ScenariosRevisionAgent:
#     pass
#
#
# class ActorsGenerationAgent:
#     pass
#
#
# class ActorsRevisionAgent:
#     pass
#
#
# class UseCasesGenerationAgent:
#     pass
#
#
# class UseCasesRevisionAgent:
#     pass
#
#
# class ClassDiagramGenerationAgent:
#     pass
#
#
# class ClassDiagramRevisionAgent:
#     pass
#
#
# class SeqDiagramGenerationAgent:
#     pass
#
#
# class SeqDiagramRevisionAgent:
#     pass
#
#
# class JudgeAgent:
#     pass


from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Type

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
