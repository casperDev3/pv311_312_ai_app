from fastapi import APIRouter, status
from datetime import datetime
from models import (
    HealthResponse
)

# init
router = APIRouter()


@router.get('/health/', response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health():
    return HealthResponse(
        status=200,
        success=True,
        data="Servers are working!",
        last_check=datetime.utcnow()
    )
