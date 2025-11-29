import uvicorn
from fastapi import FastAPI
from controllers import (
    default,
    users,
    home
)
from constants.collection_name import USERS

# init
app = FastAPI(title="FastApi JWT", version="0.0.1")

# routing
app.include_router(home, prefix="", tags=["home"])
app.include_router(default, prefix="/api", tags=["default"])
app.include_router(users, prefix=f"/api/{USERS}", tags=[f"{USERS}"])


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)


if __name__ == "__main__":
    main()
