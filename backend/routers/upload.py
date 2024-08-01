from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, HTTPException, Query
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


HABR_START = "https://habr.com"
PAGE = "/page"

def get_article_data(article_link):
    try:
        article_r = requests.get(article_link)
        article_soup = BeautifulSoup(article_r.content, features="html.parser")

        title = article_soup.find("h1", "tm-article-snippet__title").find("span").getText()
        content_blocks = article_soup.find_all("div", class_="tm-article-body__block")
        content = "\n".join([block.get_text(strip=True) for block in content_blocks])
        summary = content_blocks[0].text.strip() if content_blocks else ""

        return {
            "title": title,
            "link": article_link,
            "summary": summary,
            "content": content
        }
    except Exception as e:
        print(f"Error parsing article {article_link}: {e}")
        return None

@router.post("/parse_and_save")
async def parse_and_save(
    topic: str, 
    start_page: int = Query(1, ge=1), 
    end_page: int = Query(1, ge=1), 
    db: AsyncSession = Depends(get_db_connection)
):
    articles = []
    for page_number in range(start_page, end_page + 1):
        url = f"https://habr.com/ru/search/?q={topic}&target_type=posts&order=relevance&page={page_number}"
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Error fetching data from Habr")

        soup = BeautifulSoup(response.content, "html.parser")
        for article in soup.find_all("article", class_="tm-articles-list__item"):
            title_tag = article.find("a", class_="tm-article-snippet__title-link")
            if not title_tag:
                continue

            link = "https://habr.com" + title_tag["href"]

            article_data = get_article_data(link)
            if article_data:
                articles.append(Article(**article_data))

    if not articles:
        raise HTTPException(status_code=404, detail="No articles found")

    async with db.begin():
        db.add_all(articles)
        await db.commit()

    return {"message": f"Articles from page {start_page} to {end_page} parsed and saved successfully"}


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