from fastapi import APIRouter
from models.scheme import StatusCode


router = APIRouter()


@router.get("/", response_model=StatusCode)
def index():
    return {"code": 200, "status": "service is UP"}


@router.get("/healcheck", response_model=StatusCode)
def index():
    return {"code": 200, "status": "OK"}
