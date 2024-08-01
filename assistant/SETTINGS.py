from dotenv import load_dotenv
import os
HOST = 'ch_server'
PORT = "8123"
TABLE_NAME = "Data"
DEVICE = "cuda"
LLM_PATH = "model-q8_0.gguf"
MODEL_EMB_NAME = "ai-forever/sbert_large_nlu_ru"
SYSTEM_PROMPT = """
Эта система предназначена для предоставления помощи организаторам и участникам хакатонов. Система получает контекст из данных о хакатонах и запрос пользователя, и должна предоставить точный и информативный анализ.

Инструкции:
1. Обработай входные данные и обеспечь правильное понимание контекста.
2. Проанализируй запрос пользователя, используя предоставленный контекст.
3. Предоставь четкий и полный анализ, относящийся к запросу пользователя.
4. Используй информацию из контекста для формирования ответа.

КОНТЕКСТ: {}
ЗАПРОС: {}

"""

SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13
ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}
load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
