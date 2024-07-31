from fastapi import FastAPI
from routers import upload, index
from models.database import init_db
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="РОСДОРНИИ")
app.include_router(index.router, tags=["healcheck"])
app.include_router(upload.router, prefix="/upload")

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    await init_db()
