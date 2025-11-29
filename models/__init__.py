from .basics_models import (
    User,
    AuthUser,
    UserRegister,
    UserLogin,
    HealthResponse,
    RegisterResponse,
    LoginResponse,
    UserResponse
)

__all__ = {
    "basics_models": {
        "collections": ["User", "AuthUser"],
        "requests": ["UserRegister", "UserLogin"],
        "responses": ["HealthResponse", "RegisterResponse", "LoginResponse", "UserResponse"]
    }
}
