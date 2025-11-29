import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional, Any

# custom
from models import (
    HealthResponse,
    RegisterResponse,
    UserRegister,
)

from utils import (
    get_current_user,
    get_password_hash,
    verify_password,
    verify_token,
    create_access_token
)

# init
app = FastAPI(title="FastApi JWT", version="0.0.1")


# default endpoints
@app.get("/", response_model=HealthResponse, status_code=200)
async def root():
    return HealthResponse(
        status=200,
        success=True,
        data="It's home page!",
        last_check=datetime.utcnow()
    )


@app.get('/api/health/', response_model=HealthResponse, status_code=200)
async def health():
    return HealthResponse(
        status=200,
        success=True,
        data="Servers are working!",
        last_check=datetime.utcnow()
    )


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)


if __name__ == "__main__":
    main()
