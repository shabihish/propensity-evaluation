import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class LiteLlmClient:
    def __init__(self, model_name:  str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def get_completion(self, messages: list[dict], **kwargs ) -> dict:

        params ={}
        params.update(kwargs)

        if "gemini" in self.model_name.lower():
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["safety_settings"] = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response
    
    # TODO: Add n completions
            