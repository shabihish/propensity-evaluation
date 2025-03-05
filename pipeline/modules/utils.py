import json
import os
import re

from agents.agent import Agent
from utils.litellm_utils import get_response_content


def save_to_disk(data: dict, file_path: str):
    # make the necessary dirs
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


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
    pattern = rf"{delimiter.format(key)}\s*?fields\s*:(.*?)\n(.*?)(?={delimiter.format(r'.*?')}|\Z)"
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


def load_from_disk(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def run_agent_query(prompt, agent: Agent, logger, n_retries=5, to_json=False, json_transform_keys: list = None):
    num_retries = n_retries
    # while num_retries >= 0:
    #     try:
    result = agent(prompt)

    response = get_response_content(result, to_json=to_json)
    if to_json:
        assert len(json_transform_keys) == 2
        response = response[json_transform_keys[0]]
        response = json_obj_list_to_dict(response, json_transform_keys[1])
    return response
    #     except Exception as e:
    #         logger.error(f"Error in run_agent_query: {e}")
    #         num_retries -= 1
    # raise RuntimeError(f"Failed to run LLM query after {n_retries + 1} tries.")


def check_for_missing_fields(input_dict: dict, required_fields: list) -> dict:
    out = {}
    for obj in input_dict.values():
        missing_fields = []
        for required_field in required_fields:
            if required_field not in obj:
                missing_fields.append(required_field)
        if missing_fields:
            out[obj['name']] = {'name': obj['name'], 'missing_fields': missing_fields}
    return out


def capitalize_first_letter(text):
    return ' '.join(word[0].upper() + word[1:] for word in text.split())


def normalize_string(text, lower=True, remove_special_chars_punctuation=False):
    # Convert to lowercase
    text = text.lower()
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[-_]', ' ', text)

    # Optionally, remove special characters or punctuation
    if remove_special_chars_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text
