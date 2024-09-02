import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from insight_engine.api.routers import root

load_dotenv()
IE_TOKEN = os.environ["IE_API_TOKEN"]
security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Literal[True]:
    token = credentials.credentials
    if token != IE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")
    return True


app = FastAPI(dependencies=[Depends(verify_token)])
app.include_router(root)
