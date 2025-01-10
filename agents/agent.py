import litellm
from pydantic import BaseModel
from typing import Any, Dict, List
from agents.api_conf import APIConfiguration
from agents.output_schemas import generate_pydantic_model

litellm.set_verbose = False

import os
import time
import fcntl
import json
from threading import Thread


class RateLimiter:
    def __init__(self, max_calls_per_minute, semaphore_file):
        self.max_calls_per_minute = max_calls_per_minute
        self.semaphore_file = semaphore_file
        self._initialize_semaphore()
        self._start_release_thread()

    def _initialize_semaphore(self):
        os.makedirs(os.path.dirname(self.semaphore_file), exist_ok=True)
        if not os.path.exists(self.semaphore_file):
            with open(self.semaphore_file, 'w') as f:
                json.dump([], f)

    def _start_release_thread(self):
        def release_semaphore():
            while True:
                time.sleep(1)
                self._release_expired_calls()

        release_thread = Thread(target=release_semaphore, daemon=True)
        release_thread.start()

    def _release_expired_calls(self):
        current_time = time.time()
        with open(self.semaphore_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            call_times = json.load(f)
            call_times = [t for t in call_times if current_time - t < 60]
            f.seek(0)
            json.dump(call_times, f)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)

    def acquire(self):
        while True:
            with open(self.semaphore_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                current_time = time.time()
                call_times = json.load(f)
                call_times = [t for t in call_times if current_time - t < 60]
                if len(call_times) < self.max_calls_per_minute:
                    call_times.append(current_time)
                    f.seek(0)
                    json.dump(call_times, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return True
                fcntl.flock(f, fcntl.LOCK_UN)
            time.sleep(1)


def get_fields_without_defaults(model: BaseModel) -> List[str]:
    fields_without_defaults = []
    for field_name, field_info in model.__fields__.items():
        if field_info.default is None and field_info.default_factory is None:
            fields_without_defaults.append(field_name)
    return fields_without_defaults


class Agent:
    rate_limiter = RateLimiter(max_calls_per_minute=15, semaphore_file='.tmp/api_rate_limit.semaphore')

    def __init__(self, api_conf: APIConfiguration, sys_prompt: str = None, output_schema=None,
                 temperature: float = 1):
        # assert sys_prompt_info is not None and isinstance(sys_prompt_info, dict) and len(sys_prompt_info) == 2
        # fields, prompt = sys_prompt_info["fields"], sys_prompt_info["prompt"]
        self.system_prompt = sys_prompt if sys_prompt is not None else ""
        # self.system_prompt_fields = fields

        self._model_name = api_conf.model_name
        self._model_provider = api_conf.model_provider
        self.api_base = api_conf.api_base
        self.use_cache = api_conf.use_cache

        assert output_schema is None or isinstance(output_schema, BaseModel) or isinstance(output_schema, dict)
        self.output_schema = output_schema
        print(self.output_schema)
        # if isinstance(output_schema, dict):
        #     self.output_schema = generate_pydantic_model('default', output_schema)
        self.output_schema = {"type": "json_schema", "json_schema": {"schema": output_schema, "strict": True}}

        self.temperature = temperature

    def __call__(self, query):
        Agent.rate_limiter.acquire()

        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        completion_args = {
            "model": self._model_name,
            "custom_llm_provider": self._model_provider,
            "messages": messages,
            "temperature": self.temperature,
            "caching": self.use_cache,
        }

        if self.api_base is not None:
            completion_args["api_base"] = self.api_base

        if self.output_schema is not None:
            print('Using output schema')
            completion_args["response_format"] = self.output_schema

        print(self.output_schema)
        # exit()
        result = litellm.completion(**completion_args)

        return result
