from sys import exception
from configs import LLMConfig
import requests
import os
import openai
from openai import OpenAI
from utils import setup_logging
from typing import Union, List
import vllm
from transformers import AutoTokenizer, AutoModel

logger = setup_logging()

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("You can't directly use the meta class. Subclasses must implement this method")


class vLLM(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.url
        self.model_name = config.model_name

    def generate(self, user_input:Union[dict, str, List[dict]], sys_prompt:Union[dict, str] = None) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        messages = []
        if isinstance(sys_prompt, dict):
            messages.append(sys_prompt)
        elif isinstance(sys_prompt, str):
            messages.append({"role": "system", "content": sys_prompt})
        if isinstance(user_input, dict):
            messages.append(user_input)
        elif isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        elif isinstance(user_input, list) and all(isinstance(item, dict) for item in user_input):
            messages.extend(user_input)
        else:
            raise ValueError(f"Invalid user input type: {type(user_input)}")
        data = {
            "model": self.model_name,
            "messages": messages,
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            return content.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error: Network request failed: {e}")
            return "ERROR: THE MODEL CANNOT PROCESS THE REQUEST."
        except (KeyError, IndexError) as e:
            logger.error(f"Error: Could not parse the response JSON: {e}")
            logger.error(f"Received response: {response.text}")
            return "ERROR: THE MODEL CANNOT PROCESS THE REQUEST."
        
class GPT(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.url
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = config.model_name
        self.client = OpenAI(base_url=config.url)

    def generate(self, user_input:Union[dict, str, List[dict]], sys_prompt:Union[dict, str] = None, **kwargs) -> str:
        messages = []
        if isinstance(sys_prompt, dict):
            messages.append(sys_prompt)
        elif isinstance(sys_prompt, str):
            messages.append({"role": "system", "content": sys_prompt})
        if isinstance(user_input, dict):
            messages.append(user_input)
        elif isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        elif isinstance(user_input, list) and all(isinstance(item, dict) for item in user_input):
            messages.extend(user_input)
        else:
            raise ValueError(f"Invalid user input type: {type(user_input)}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 1024),
                stream=False
            )
        except Exception as e:
            status_code = getattr(e, "status_code", "unknown")
            logger.warning(f"API request failed with status code {status_code}: {e}", exc_info=True)
            return ""
        # logger.info(f"Response tokens: {response.usage.completion_tokens}\nPrompt tokens: {response.usage.prompt_tokens}")
        return response.choices[0].message.content
    

class TransformerLLM(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.device = config.device
        self.model_name = config.model_name
        self.tokenizer = AutoTokenizer(self.model_name)
        self.model = AutoModel(self.model_name)