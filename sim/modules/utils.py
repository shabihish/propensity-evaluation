import json
import re
import os


def fill_prompt(prompt: str, fields: list, context: dict):
    for field in fields:
        if field not in context:
            raise ValueError(f"Field {field} not found in context")

        prompt = prompt.replace('{' + f'{field}' + '}', str(context[field]))
    return prompt


def read_prompts(file_path: str, key: str, delimiter: str = r'\[{}\]', context: dict = None, logger=None):
    if context is None:
        context = dict()
    with open(file_path, 'r') as file:
        content = file.read()

    key = key.strip().lower()

    # Match everything from the key until the next delimiter or end of file
    assert '(' not in key and ')' not in key, "Key should not contain parentheses"
    pattern = rf'{delimiter.format(key)}\s*?fields\s*:(.*?)\n(.*?)(?={delimiter.format(r'.*?')}|\Z)'
    pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)

    if not matches:
        raise ValueError(f"Key {key} not found in prompts file {file_path}")

    assert len(matches) == 1, f"Key {key} not found in prompts file {file_path}"
    assert len(matches[0]) == 2, f'Pattern match failed: {matches[0]}'

    fields_line, text = matches[0][0], matches[0][1]
    fields = [x.strip().lower() for x in fields_line.split(',')]
    fields = [x for x in fields if x]  # Remove empty strings
    prompt = text.strip()

    # if logger is not None:
    #     logger.debug(
    #         f'For file: {file_path}, key: {key}, \nLoaded fields: {fields}, \nProvided context: {context},\nLoaded prompt: {prompt}\n\n')
    filled_prompt = fill_prompt(prompt, fields, context)

    return filled_prompt


def extract_required_fields(json_schema: dict) -> list:
    required_fields = []

    def traverse(schema):
        if 'required' in schema:
            required_fields.extend(schema['required'])
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                traverse(value)
        if 'items' in schema:
            traverse(schema['items'])

    traverse(json_schema)
    return required_fields


def load_output_schemas(path: str):
    with open(path, 'r') as f:
        out = json.load(f)

    return out


def json_obj_list_to_dict(json_list: list, transformation_key: str):
    out = dict()
    for obj in json_list:
        key = obj[transformation_key]
        out[key] = obj
    return out


def save_to_disk(data: dict, file_path: str):
    # make the necessary dirs
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_disk(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
