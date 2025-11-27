import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional
import uvicorn


@strawberry.type
class User:
    id: int
    name: str
    email: str
    age: Optional[int] = None


@strawberry.type
class Post:
    id: int
    title: str
    content: str
    user_id: int

@strawberry.type()
class Meta:
    total: int

@strawberry.type()
class ResponsePosts:
    status: int
    success: bool
    data: List[Post]
    # meta: Meta
    total: int

@strawberry.type()
class ResponsePost:
    status: int
    success: bool
    data: Post
    # meta: Meta
    total: int


# Фейкові дані
users_db = [
    User(id=1, name="Анна", email="anna@example.com", age=25),
    User(id=2, name="Петро", email="petro@example.com", age=30),
    User(id=3, name="Марія", email="maria@example.com")
]

posts_db = [
    Post(id=1, title="Мій перший пост", content="Привіт світ!", user_id=1),
    Post(id=2, title="Ще один пост", content="Це другий пост", user_id=1),
    Post(id=3, title="Python", content="Python - чудова мова", user_id=2)
]


@strawberry.type
class Query:
    @strawberry.field()
    def health(self) -> str:
        return "Server works good!"

    @strawberry.field()
    def posts(self)-> ResponsePosts:
        return ResponsePosts(
            status=200,
            success=True,
            data=posts_db,
            total=len(posts_db)
        )

    @strawberry.field()
    def one_post(self, id: int)->ResponsePost:
        return ResponsePost(
            success=True,
            status=200,
            data=posts_db[id - 1],
            total=len(posts_db)
        )

@strawberry.type
class Mutation:
    @strawberry.mutation()
    def create_post(self, title: str) -> Post:
        new_id = len(posts_db) + 1
        new_post = Post(id=new_id, title=title, content="", user_id=1)
        posts_db.append(new_post)
        return new_post

schema = strawberry.Schema(query=Query, mutation=Mutation)
app = FastAPI(title="Test GraphQL")

app.include_router(GraphQLRouter(schema), prefix="/graphql")

@app.get("/rest/health")
def check_health():
    return {
        "message": "Server works good! REST API"
    }

@app.get("/rest/posts/")
def get_all_posts():
    return {
        "status": 200,
        "success": True,
        "data": posts_db,
        "meta": {
            "total": len(posts_db)
        }
    }

@app.get("/rest/posts/{id}")
def get_all_posts(requests):
    # id = requests
    return {
        "status": 200,
        "success": True,
        "data": posts_db,
        "meta": {
            "total": len(posts_db)
        }
    }

@app.get("/")
def root():
    return {
        "status": 200,
        "success": True
    }


def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000
    )


if __name__ == "__main__":
    main()
