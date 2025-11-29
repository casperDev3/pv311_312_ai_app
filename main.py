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
    AuthUser,
    User,
    LoginResponse,
    UserLogin,
    UserResponse
)

from utils import (
    get_current_user,
    get_password_hash,
    verify_password,
    verify_token,
    create_access_token
)

from db.users import fake_users_db
from constants.keys import ACCESS_TOKEN_EXPIRE_MINUTES

# init
app = FastAPI(title="FastApi JWT", version="0.0.1")


# default endpoints
@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def root():
    return HealthResponse(
        status=200,
        success=True,
        data="It's home page!",
        last_check=datetime.utcnow()
    )


@app.get('/api/health/', response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health():
    return HealthResponse(
        status=200,
        success=True,
        data="Servers are working!",
        last_check=datetime.utcnow()
    )


@app.post("/api/users/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="username already exist")

    hashed_password = get_password_hash(user.password)
    fake_users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    return RegisterResponse(
        status=201,
        success=True,
        data=User(
            username=user.username,
            email=user.email,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    )


@app.post("/api/users/login/", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(user_credentials: UserLogin):
    user = fake_users_db.get(user_credentials.username)
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return LoginResponse(
        success=True,
        status=status.HTTP_200_OK,
        data=AuthUser(
            jwt=access_token,
            token_type="Bearer",
            expires=ACCESS_TOKEN_EXPIRE_MINUTES,
            user=User(
                username=user["username"],
                email=user["email"],
                created_at=[user["created_at"]],
                updated_at=[user["updated_at"]]
            )
        )
    )


@app.get("/api/users/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_info(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        success=True,
        status=status.HTTP_200_OK,
        data=User(
            username=current_user["username"],
            email=current_user["email"],
            created_at=current_user["created_at"],
            updated_at=current_user["updated_at"]
        )
    )



def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)


if __name__ == "__main__":
    main()
