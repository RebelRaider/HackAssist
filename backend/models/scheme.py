from pydantic import BaseModel
from typing import Optional

class StatusCode(BaseModel):
    code: int
    status: str


class OKSItemSchema(BaseModel):
    L3: Optional[str]
    L4: Optional[int]
    L5: Optional[int]
    L6: Optional[int]
    L7: Optional[int]
    L8: Optional[int]
    L9: Optional[int]
    L10: Optional[int]
    code: str
    category_4: Optional[str]
    category_5: Optional[str]
    level_1: Optional[str]
    level_2: Optional[str]
    level_3: Optional[str]
    level_4: Optional[str]
    level_5: Optional[str]
    date_added: Optional[str]
    date_modified: Optional[str]
    data_source: Optional[str]

    class Config:
        orm_mode = True