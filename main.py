from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import requests
import base64

app = FastAPI(title="OpenAI API Integration", version="0.0.1")

class ChatRequests(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    max_token: int = 150

class ImageRequests(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1

openai.api_key = os.getenv("OPENAI_KEY_API")

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()