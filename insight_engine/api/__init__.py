from fastapi import FastAPI

from insight_engine.api.routers import root

app = FastAPI()
app.include_router(root)
