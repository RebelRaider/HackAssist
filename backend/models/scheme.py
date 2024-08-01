from pydantic import BaseModel

class StatusCode(BaseModel):
    code: int
    status: str
