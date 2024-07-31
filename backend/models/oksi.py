from .database import Base
from sqlalchemy import Column, String, Integer, DateTime, func, Float, ForeignKey
from sqlalchemy.orm import relationship


class OKSItem(Base):
    __tablename__ = "oksi_items"
    L3 = Column(String, nullable=True, default="")
    L4 = Column(Integer, nullable=True, default=0)
    L5 = Column(Integer, nullable=True, default=0)
    L6 = Column(Integer, nullable=True, default=0)
    L7 = Column(Integer, nullable=True, default=0)
    L8 = Column(Integer, nullable=True, default=0)
    L9 = Column(Integer, nullable=True, default=0)
    L10 = Column(Integer, nullable=True, default=0)
    code = Column(String, nullable=False, primary_key=True, index=True)
    category_4 = Column(String, nullable=True, default="")
    category_5 = Column(String, nullable=True, default="")
    level_1 = Column(String, nullable=True, default="")
    level_2 = Column(String, nullable=True, default="")
    level_3 = Column(String, nullable=True, default="")
    level_4 = Column(String, nullable=True, default="")
    level_5 = Column(String, nullable=True, default="")
    date_added = Column(DateTime, nullable=True, default=func.min())
    date_modified = Column(DateTime, nullable=True, default=func.min())
    data_source = Column(String, nullable=True, default="")


class Section(Base):
    __tablename__ = 'section'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    code = Column(String)
    parent_id = Column(Integer, ForeignKey('section.id'))  # Ссылка на родительский раздел
    works = relationship("Work", back_populates="section", lazy='selectin')
    async def to_dict(self):
        works_dicts = [await work.to_dict() for work in self.works] if self.works else None
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "code": self.code,
            "parent_id": self.parent_id,
            "works": works_dicts
        }

class Work(Base):
    __tablename__ = 'work'

    id = Column(Integer, primary_key=True)
    code = Column(String)
    end_name = Column(String)
    measure_unit = Column(String)
    content = Column(String)
    section_id = Column(Integer, ForeignKey('section.id'))  # Ссылка на раздел
    section = relationship("Section", back_populates="works", lazy='selectin')
    resources = relationship("WorkResource", back_populates="work", lazy='selectin')

    async def to_dict(self):
        res_dict = [await work_resource.to_dict() for work_resource in self.resources]  if self.resources else None
        return {
            "id": self.id,
            "code": self.code,
            "end_name": self.end_name,
            "measure_unit": self.measure_unit,
            "content": self.content,
            "section_id": self.section_id,
            "resources": res_dict
        }

class Resource(Base):
    __tablename__ = 'resource'

    id = Column(Integer, primary_key=True)
    code = Column(String)
    name = Column(String)
    cost = Column(Float)
    opt_cost = Column(Float)
    measure_unit = Column(String)

    async def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "cost": self.cost,
            "opt_cost": self.opt_cost,
            "measure_unit": self.measure_unit
        }

class WorkResource(Base):
    __tablename__ = 'work_resource'

    id = Column(Integer, primary_key=True)
    work_id = Column(Integer, ForeignKey('work.id'))
    resource_id = Column(Integer, ForeignKey('resource.id'))
    quantity = Column(Float)
    work = relationship("Work", back_populates="resources", lazy='selectin')
    resource = relationship("Resource", lazy='selectin')

    async def to_dict(self):
        res = await self.resource.to_dict() if self.resource else None
        return {
            "id": self.id,
            "work_id": self.work_id,
            "resource_id": self.resource_id,
            "quantity": self.quantity,
            "resource": res
        }