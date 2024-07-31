from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, HTTPException
import requests
from models.article import Article
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import get_db_connection
from sqlalchemy.future import select
import clickhouse_connect
from utils.ml import drop_table, create_table, txt2embeddings, load_models
from SETTINGS import HOST, PORT, TABLE_NAME, MODEL_EMB_NAME, DEVICE
import re
router = APIRouter(tags=["upload"])


@router.post("/parse_and_save")
async def parse_and_save(topic: str, db: AsyncSession = Depends(get_db_connection)):
    url = f"https://habr.com/ru/search/?q={topic}&target_type=posts&order=relevance"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Error fetching data from Habr")

    soup = BeautifulSoup(response.content, "html.parser")
    articles = []

    for article in soup.find_all("article", class_="post"):
        title = article.find("a", class_="post__title_link").text
        link = article.find("a", class_="post__title_link")["href"]
        summary = article.find("div", class_="post__text").text.strip()
        content = ""  # you may want to fetch full content separately

        articles.append(Article(title=title, link=link, summary=summary, content=content))

    async with db.begin():
        db.add_all(articles)
        await db.commit()
    
    return {"message": "Articles parsed and saved successfully"}

@router.patch("/update_RAG")
async def append_to_clickhouse(db: AsyncSession = Depends(get_db_connection)):
    async with db.begin():
        result = await db.execute(select(Article))
        articles = result.scalars().all()

    dicts = [article.__dict__ for article in articles]
    result_list = []
    for d in dicts:
        text = f"Title: {d['title']}\nSummary: {d['summary']}"
        clean_text = re.sub(r"[{}:;\'()\[\]]", "", text)
        clean_text = re.sub(r"None", "", clean_text)
        clean_text = re.sub(r"  ", " ", clean_text)
        result_list.append(clean_text)


    client = clickhouse_connect.get_client(host=HOST, port=PORT)
    drop_table(client, TABLE_NAME)
    create_table(client, TABLE_NAME)
    
    data = [{"text": text} for text in result_list]
    text_data = [item.get("text") for item in data]
    

    tokenizer, model = load_models(MODEL_EMB_NAME, device=DEVICE)
    embeddings = txt2embeddings(text_data, tokenizer, model, device=DEVICE)
    
    for i, item in enumerate(data):
        vectors = ",".join([str(float(vector)) for vector in embeddings[i]])
        query =  f"""INSERT INTO "{TABLE_NAME}"("Text", "Embedding") VALUES('{item.get('text')}', ({vectors}))"""
        client.command(query)