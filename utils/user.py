from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from utils import verify_token
from db.users import fake_users_db


security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    username = verify_token(token)
    user = fake_users_db.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Token is invalid!")
    return user