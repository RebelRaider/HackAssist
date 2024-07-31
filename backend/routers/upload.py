from fastapi import APIRouter, UploadFile, Depends, HTTPException
from models.oksi import Section, Work, Resource, WorkResource, OKSItem
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from models.database import get_db_connection
from fastapi import UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from utils.utils import parse_xlsx, parse_gsn, parse_fsnb, dict_to_str
import pandas as pd
from sqlalchemy.future import select
from models.scheme import OKSItemSchema
from typing import List
from utils.ml import *
import clickhouse_connect
from SETTINGS import HOST, PORT
router = APIRouter(tags=["upload"])


@router.post("/upload_oksi/")
async def upload_file(file: UploadFile, db: AsyncSession = Depends(get_db_connection)):
    if (
        file.content_type
        != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    df = parse_xlsx(file_location)

    async with db.begin():
        for index, row in df.iterrows():
            date_added = row["date_added"] if pd.notna(row["date_added"]) else None
            date_modified = (
                row["date_modified"] if pd.notna(row["date_modified"]) else None
            )

            item = OKSItem(
                L3=row["L3"],
                L4=row["L4"],
                L5=row["L5"],
                L6=row["L6"],
                L7=row["L7"],
                L8=row["L8"],
                L9=row["L9"],
                L10=row["L10"],
                code=row["code"],
                category_4=row["category_4"],
                category_5=row["category_5"],
                level_1=row["level_1"],
                level_2=row["level_2"],
                level_3=row["level_3"],
                level_4=row["level_4"],
                level_5=row["level_5"],
                date_added=date_added,
                date_modified=date_modified,
                data_source=row["data_source"],
            )
            db.add(item)
        await db.commit()

    return JSONResponse(
        status_code=200, content={"message": "File processed successfully"}
    )


@router.get("/items/", response_model=List[OKSItemSchema])
async def read_items(db: AsyncSession = Depends(get_db_connection)):
    async with db.begin():
        result = await db.execute(select(OKSItem))
        items = result.scalars().all()
        items_with_dates_as_str = [
            {
                **item.__dict__,
                "date_added": (
                    item.date_added.strftime("%Y-%m-%d") if item.date_added else None
                ),
                "date_modified": (
                    item.date_modified.strftime("%Y-%m-%d")
                    if item.date_modified
                    else None
                ),
            }
            for item in items
        ]

        return items_with_dates_as_str


# Эндпоинты для загрузки файлов
@router.post("/upload_gsn/")
async def upload_gsn(file: UploadFile, db: AsyncSession = Depends(get_db_connection)):
    content = await file.read()
    await parse_gsn(content, db)
    return {"status": "success"}


@router.post("/upload_fsnb/")
async def upload_fsnb(file: UploadFile, db: AsyncSession = Depends(get_db_connection)):
    content = await file.read()
    await parse_fsnb(content, db)
    return {"status": "success"}


# Эндпоинты для получения данных
@router.get("/get_gsn/")
async def get_gsn(db: AsyncSession = Depends(get_db_connection)):
    async with db.begin():
        result = await db.execute(
            select(Section).limit(20)  # Ограничение на 20 записей
        )
        sections = result.scalars().all()
        return JSONResponse(content=[await section.to_dict() for section in sections])


@router.get("/get_fsnb/")   
async def get_fsnb(db: AsyncSession = Depends(get_db_connection)):
    async with db.begin():
        result = await db.execute(
            select(Resource).limit(20)  # Ограничение на 20 записей
        )
        resources = result.scalars().all()
        return JSONResponse(
            content=[await resource.to_dict() for resource in resources]
        )

@router.patch("/update_RAG") 
async def append_to_clickhouse(db: AsyncSession = Depends(get_db_connection)):
    async with db.begin():
        result = await db.execute(
            select(Section)
        )
        sections = result.scalars().all()
        dicts = [await section.to_dict() for section in sections]
    result_list = []
    for d in dicts:
        text = dict_to_str(d)
        clean_text = re.sub(r"[{}:;\'()\[\]]", "", text)
        clean_text = re.sub(r"None", "", clean_text)
        clean_text = re.sub(r"  ", " ", clean_text)
        result_list.append(clean_text)
    TABLE_NAME = "Data"
    MODEL_EMB_NAME = "ai-forever/sbert_large_nlu_ru"
    DEVICE = "cpu"
    client = clickhouse_connect.get_client(host=HOST, port=PORT)
    drop_table(client, TABLE_NAME)
    create_table(client, TABLE_NAME)
    data = [{
        "text": text,
    } for text in result_list]
    text_data = [item.get("text") for item in data]
    tokenizer, model = load_models(MODEL_EMB_NAME, device=DEVICE)
    embeddings = txt2embeddings(text_data, tokenizer, model, device=DEVICE)
    for i, item in enumerate(data):
        vectors = ",".join([str(float(vector)) for vector in embeddings[i]])
        query =  f"""INSERT INTO "{TABLE_NAME}"("Text", "Embedding") VALUES('{item.get('text')}', ({vectors}))"""
        client.command(query)