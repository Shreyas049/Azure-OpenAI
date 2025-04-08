from openai import AzureOpenAI
from typing import Optional, Dict, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import pydantic
import json

class AzureOpenAICaller():
    def __init__(
        self,
        azure_openai_endpoint: str,
        azure_openai_key: str,
        azure_openai_deployment: str,
        azure_openai_deployment_version: str
    ):
        """
        Initialize AzureOpenAI client        
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            azure_deployment=azure_openai_deployment,
            api_version=azure_openai_deployment_version
        )

        self.model = azure_openai_deployment

        # usage trackers
        self.model_name = ""
        self.tokens_prompt = 0
        self.tokens_completion = 0

    
    RETRY_DECORATOR = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=10)
    )

    
    def _update_usage_metrics(self, completion: dict) -> Dict[str, Union[int, str]]:
        """Helper method to update usage metrics and return usage dict."""
        self.tokens_prompt += completion['usage']['prompt_tokens']
        self.tokens_completion += completion['usage']['completion_tokens']
        self.model_name = completion['model']
        return {
            "prompt_tokens": self.tokens_prompt,
            "completion_tokens": self.tokens_completion,
            "model": self.model_name
        }
    
    
    @RETRY_DECORATOR
    async def call_openai_chat(
        self,
        input_text: str,
        system_prompt: str
    ) -> Tuple[dict, dict]:
        """
        Call OpenAI chat API for general extraction using a system prompt.

        Args:
            input_text: Text to process and extract information from
            system_prompt: Prompt to guide model's behavior (must include 'json')
        Returns:
            Tuple of (response dict, usage dict)
        """
        final_query = f"Extract info from following Context:\n{input_text}"
        completion = await self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_query}
            ],
            model=self.model
        )
        completion_data = completion.model_dump()
        
        response = json.loads(completion_data['choices'][0]['message']['content'])
        usage = self._update_usage_metrics(completion_data)
        return response, usage

    
    @RETRY_DECORATOR
    async def call_openai_json_extraction(
        self,
        input_text: str,
        system_prompt: str,
        extract_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """
        Call OpenAI API for JSON-structured extraction using either extract_prompt or json_schema.

        Args:
            input_text: Text to process and extract information from
            system_prompt: Prompt to guide model's behavior (must include 'json')
            extract_prompt: Optional specific extraction prompt
            json_schema: Optional JSON schema for structured output
        Raises:
            Exception: If neither or both extract_prompt and json_schema are provided
        Returns:
            Tuple of (response dict, usage dict)
        """
        if (extract_prompt and json_schema) or (not extract_prompt and not json_schema):
            raise ValueError("Must provide exactly one of: extract_prompt or json_schema")

        final_query = (
            f"{extract_prompt}\nExtract in JSON format from Input-Context:\n{input_text}"
            if extract_prompt
            else f"Extract in JSON format from Input-Context:\n{input_text}"
        )

        response_format = {"type": "json_object"} if extract_prompt else {"type": "json_schema", "json_schema": json_schema}

        completion = await self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_query}
            ],
            model=self.model,
            response_format=response_format
        )
        completion_data = completion.model_dump()
        
        response = json.loads(completion_data['choices'][0]['message']['content'])
        usage = self._update_usage_metrics(completion_data)
        return response, usage

    
    @RETRY_DECORATOR
    async def call_openai_pydantic_extraction(
        self,
        input_text: str,
        system_prompt: str,
        response_model: pydantic.BaseModel
    ) -> Tuple[dict, dict]:
        """
        Call OpenAI API for extraction using Pydantic model structure.

        Args:
            input_text: Text to process and extract information from
            system_prompt: Prompt to guide model's behavior
            response_model: Pydantic model defining response structure
        Returns:
            Tuple of (response dict, usage dict)
        """
        final_query = f"Extract the pydantic information from below context.\nContext: {input_text}"
        
        completion = await self.client.beta.chat.completions.parse(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_query}
            ],
            model=self.model,
            response_format=response_model
        )
        completion_data = completion.model_dump()
        
        response = completion_data['choices'][0]['message']['parsed']
        usage = self._update_usage_metrics(completion_data)
        return response, usage
