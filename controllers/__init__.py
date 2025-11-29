from .default import router as default
from .users import router as users
from .home import router as home

__all__ = {
    "collections": ["default", "users", "home"]
}