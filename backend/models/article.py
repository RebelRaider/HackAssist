from .database import Base
from sqlalchemy import Column, Integer, String, Text

class Article(Base):
    __tablename__ = 'articles'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    link = Column(String, unique=True, index=True)
    summary = Column(Text)
    content = Column(Text)
