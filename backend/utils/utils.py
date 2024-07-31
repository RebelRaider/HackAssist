import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.oksi import Section, Work, WorkResource, Resource
import xml.etree.ElementTree as ET

def parse_xlsx(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
        "L9",
        "L10",
        "code",
        "category_4",
        "category_5",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "date_added",
        "date_modified",
        "data_source",
    ]
    df.fillna("", inplace=True)
    return df


async def parse_gsn(content: bytes, db: AsyncSession):
    root = ET.fromstring(content)
    
    async def add_section(element, parent_section=None):
        section_name = element.attrib.get("Name")
        section_type = element.attrib.get("Type")
        section_code = element.attrib.get("Code")

        section = Section(name=section_name, type=section_type, code=section_code, parent_id=parent_section.id if parent_section else None)
        db.add(section)
        await db.flush()  # To get section.id

        for child in element:
            if child.tag == "Section":
                await add_section(child, section)
            elif child.tag == "NameGroup":
                for work_elem in child.findall("Work"):
                    await add_work(work_elem, section)
    
    async def add_work(element, section):
        work_code = element.attrib.get("Code")
        work_end_name = element.attrib.get("EndName")
        work_measure_unit = element.attrib.get("MeasureUnit")
        work_content = "\n".join([item.attrib.get("Text") for item in element.find("Content").findall("Item")])

        work = Work(code=work_code, end_name=work_end_name, measure_unit=work_measure_unit, content=work_content, section_id=section.id)
        db.add(work)
        await db.flush()  # To get work.id

        resources_elem = element.find("Resources")
        if resources_elem is not None:
            for resource_elem in resources_elem.findall("Resource"):
                await add_work_resource(resource_elem, work)

    async def add_work_resource(element, work):
        resource_code = element.attrib.get("Code")
        resource_end_name = element.attrib.get("EndName")
        try:
            resource_quantity = float(element.attrib.get("Quantity"))
        except ValueError:
            resource_quantity = 0

        stmt = select(Resource).filter_by(code=resource_code)
        result = await db.execute(stmt)
        resource = result.scalars().first()

        if resource is None:
            resource = Resource(code=resource_code, name=resource_end_name)
            db.add(resource)
            await db.flush()  # To get resource.id

        work_resource = WorkResource(work_id=work.id, resource_id=resource.id, quantity=resource_quantity)
        db.add(work_resource)
    
    resources_dir = root.find("ResourcesDirectory")
    if resources_dir is not None:
        for section_elem in resources_dir.findall("ResourceCategory/Section"):
            await add_section(section_elem)

    await db.commit()


async def parse_fsnb(file_contents, session: AsyncSession):
    tree = ET.ElementTree(ET.fromstring(file_contents))
    root = tree.getroot()

    resources_directory = root.find("ResourcesDirectory")
    for category in resources_directory.findall("ResourceCategory"):
        for section in category.findall("Section"):
            await parse_section_fsnb(section, session)


async def parse_section_fsnb(section, session: AsyncSession):
    for resource in section.findall("Resource"):
        code = resource.get("Code")
        name = resource.get("Name")
        measure_unit = resource.get("MeasureUnit")
        prices = resource.find("Prices")
        cost = float(prices.find("Price").get("Cost")) if prices.find("Price").get("Cost") else None
        opt_cost = float(prices.find("Price").get("OptCost")) if prices.find("Price").get("OptCost") else None

        new_resource = Resource(
            code=code,
            name=name,
            measure_unit=measure_unit,
            cost=cost,
            opt_cost=opt_cost,
        )
        session.add(new_resource)
        await session.commit()

    for subsection in section.findall("Section"):
        await parse_section_fsnb(subsection, session)


def dict_to_str(d):
    csv_string = ""
    for key, value in d.items():
        if isinstance(value, dict):
            csv_string += dict_to_str(value)
        else:
            csv_string += str(value) + ","
    return csv_string
