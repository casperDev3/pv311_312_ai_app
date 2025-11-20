from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os
import requests
import uvicorn

app = FastAPI(title="OpenAI API Integration", version="0.0.1")
token = os.getenv("OPENAI_KEY_API")
client = OpenAI(api_key=token)

class ChatRequests(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    max_token: int = 150

class ImageRequests(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1

# openai.api_key = os.getenv("OPENAI_KEY_API")

@app.get('/')
async def root():
    return {
        "status_code": 200,
        "success": True,
        "data": {},
        "message": "OpenAI сервер працює!"
    }

@app.get('/api/health')
async def check_health():
    return {
        "status_code": 200,
        "success": True,
        "data": {},
        "message": "Сервер працює справно!"
    }

@app.post('/api/chat/')
async def ask_chat(request: ChatRequests):
    try:
        response = client.responses.create(
            model= request.model,
            input = [
                {
                    "role": "user",
                    "content": request.message
                }
            ],
        )

        answer = response.output[0].content[0].text
        token_used = response.usage

        return {
            "status_code": 200,
            "success": True,
            "data": answer,
            "meta": {
                "model": request.model,
                "response": response,
                "tokens_used": token_used
            }
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error: {str(err)}")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()