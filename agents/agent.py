import litellm
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from typing import Any, Dict, List
from agents.api_conf import APIConfiguration

litellm.set_verbose = False

import os
import time
import fcntl
import json
from threading import Thread
from litellm import RateLimitError

class RateLimiter:
    def __init__(self, api_keys: list[str], model_names: list[str], max_calls_per_minute: int, semaphore_dir: str):
        self.api_keys = api_keys
        self.model_names = model_names
        self.max_calls_per_minute = max_calls_per_minute
        self.semaphore_dir = semaphore_dir
        self.current_key_index = 0
        self.current_model_index = 0
        self._initialize_semaphores()
        self._start_release_thread()

    def _initialize_semaphores(self):
        os.makedirs(self.semaphore_dir, exist_ok=True)
        for key in self.api_keys:
            for model in self.model_names:
                semaphore_file = self._get_semaphore_file(key, model)
                if not os.path.exists(semaphore_file):
                    with open(semaphore_file, 'w') as f:
                        json.dump([], f)

    def _get_semaphore_file(self, key: str, model: str) -> str:
        return os.path.join(self.semaphore_dir, f"{key}_{model}_semaphore.json")

    def _start_release_thread(self):
        def release_semaphore():
            while True:
                time.sleep(1)
                self._release_expired_calls()

        release_thread = Thread(target=release_semaphore, daemon=True)
        release_thread.start()

    def _release_expired_calls(self):
        current_time = time.time()
        tot_active_sessions = 0
        for key in self.api_keys:
            for model in self.model_names:
                semaphore_file = self._get_semaphore_file(key, model)
                with open(semaphore_file, 'r+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    call_times = json.load(f)
                    call_times = [entry for entry in call_times if current_time - float(entry.split(":")[0]) < 60]
                    tot_active_sessions += len(call_times)
                    f.seek(0)
                    json.dump(call_times, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
        if int(current_time) % 5 == 0:
            print(f"Active sessions: {tot_active_sessions}")

    def _set_api_key(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    def switch_key_and_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.model_names)
        if self.current_model_index == 0:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._set_api_key(self.api_keys[self.current_key_index])

    def acquire(self) -> tuple:
        failed = 0
        while True:
            current_key = self.api_keys[self.current_key_index]
            current_model = self.model_names[self.current_model_index]
            self._set_api_key(current_key)
            semaphore_file = self._get_semaphore_file(current_key, current_model)
            with open(semaphore_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                current_time = time.time()
                call_times = json.load(f)
                call_times = [entry for entry in call_times if current_time - float(entry.split(":")[0]) < 60]
                if len(call_times) < self.max_calls_per_minute:
                    call_times.append(f"{current_time}:{current_model}")
                    f.seek(0)
                    json.dump(call_times, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return (current_key, current_model, current_time)
                fcntl.flock(f, fcntl.LOCK_UN)

            # Switch to the next key and model if the current one is exhausted
            self.switch_key_and_model()
            failed += 1
            if failed > 0 and failed % (len(self.api_keys) * len(self.model_names)) == 0:
                time.sleep(3)

    def release(self, session: tuple):
        current_key, current_model, call_time = session
        semaphore_file = self._get_semaphore_file(current_key, current_model)
        with open(semaphore_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            call_times = json.load(f)
            call_times.remove(f"{call_time}:{current_model}")
            f.seek(0)
            json.dump(call_times, f)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)


def get_fields_without_defaults(model: BaseModel) -> List[str]:
    fields_without_defaults = []
    for field_name, field_info in model.__fields__.items():
        if field_info.default is None and field_info.default_factory is None:
            fields_without_defaults.append(field_name)
    return fields_without_defaults


class Agent:
    assert os.environ.get("API_KEYS") is not None, "API_KEYS environment variable is not set"
    API_KEYS = [x.strip('"') for x in os.getenv("API_KEYS").strip("()").split(" ")]
    rate_limiter = RateLimiter(api_keys=API_KEYS, model_names=['gemini-2.0-flash', 'gemini-1.5-flash'], max_calls_per_minute=15,
                               semaphore_dir='.tmp/')

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

        assert output_schema is None or isinstance(output_schema, BaseModel) or isinstance(output_schema,
                                                                                           ModelMetaclass) or isinstance(
            output_schema, dict)
        self.output_schema = output_schema
        # print(self.output_schema)
        if isinstance(output_schema, dict):
            #     self.output_schema = generate_pydantic_model('default', output_schema)
            self.output_schema = {"type": "json_schema", "json_schema": {"schema": output_schema, "strict": True}}

        self.temperature = temperature

    def __call__(self, query):
        session = None
        session_model_name = self._model_name
        if self._model_provider == "gemini":
            session = Agent.rate_limiter.acquire()
            session_model_name = session[1]

        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        completion_args = {
            "model": session_model_name,
            "custom_llm_provider": self._model_provider,
            "messages": messages,
            "temperature": self.temperature,
            "caching": self.use_cache,
        }

        if self.api_base is not None:
            completion_args["api_base"] = self.api_base

        if self.output_schema is not None:
            completion_args["response_format"] = self.output_schema

        try:
            result = litellm.completion(**completion_args)
        except RateLimitError as e:
            if session is not None:
                self.rate_limiter.release(session)
                self.rate_limiter.switch_key_and_model()
            raise

        return result
