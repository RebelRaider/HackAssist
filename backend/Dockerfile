FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]