from fastapi import APIRouter, status
from datetime import datetime
from models import (
    HealthResponse,
)

router = APIRouter()

# default endpoints
@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def root():
    return HealthResponse(
        status=200,
        success=True,
        data="It's home page!",
        last_check=datetime.utcnow()
    )
