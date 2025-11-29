from typing import Optional, Any
from pydantic import BaseModel, EmailStr, Field

# models__collections
class User(BaseModel):
    username: str
    mail: str
    created_at: Any
    updated_at: Any


class AuthUser(BaseModel):
    user: User
    jwt: str
    token_type: str
    expires: Any


# models__requests
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., max_length=6)


class UserLogin(BaseModel):
    username: str
    password: str = Field(..., max_length=6)


# models__responses
class HealthResponse(BaseModel):
    status: int
    success: bool
    data: str
    last_check: Any


class RegisterResponse(BaseModel):
    status: int
    success: bool
    data: User


class LoginResponse(BaseModel):
    status: int
    success: bool
    data: AuthUser
