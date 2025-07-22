from pydantic import BaseModel
from typing import Union,Literal
from string import Template
from langchain_openai import ChatOpenAI


class McqOutput(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class OpenAIProvider:
    
    def __init__(self, api_key: str, api_url: str = None,  temperature: float = 0.0, model:str = None):
        
        self.api_key = api_key
        self.api_url = api_url
        self.temperature = temperature
        
        self.client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                model=model,
                max_retries=3,
                temperature=self.temperature,
        )

    def generate_response(self, messages: list=[]):
        
        if not self.client:
                print("OpenAI Client not initialized")
                return None
        try:
        
            result = self.client.with_structured_output(schema=McqOutput,strict=True).invoke(messages)
            return result.answer
        
        except Exception as e:
            print(f"Error generating response using OpenAI: {e}")
            return None
    
    def construct_prompt(self, prompt: Union[str, Template], role: str, vars: dict = {}):
        if isinstance(prompt, Template):
            prompt = prompt.substitute(vars)
            
        message = {
                "role": role,
                "content": prompt
        }
    
        return message