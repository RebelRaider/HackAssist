from addons import load_models
from SETTINGS import MODEL_EMB_NAME, LLM_PATH, HOST, PORT, DEVICE, TG_TOKEN
from llama_cpp import Llama
from ml import interact_manager, request2similiars
import clickhouse_connect
from telegram.ext import Application, CommandHandler, filters, MessageHandler

tokenizer, model = load_models(MODEL_EMB_NAME, device=DEVICE)
client = clickhouse_connect.get_client(host=HOST, port=PORT)
llm_model = Llama(
    model_path=LLM_PATH,
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    n_parts=1,
)

async def start(update, context):
    await update.message.reply_text(
        "Привет! Я бот для анализа данных из ГЭСН. Задайте свой вопрос!"
    )


async def handle_message(update, context):
    question = update.message.text
    chat_id = update.message.chat_id
    context_data = request2similiars(question, tokenizer, model, client, limit=4)
    await context.bot.send_message(
            chat_id=chat_id,
            text=f"Наиболее сходие документы в базе данных:\n\n{context_data}",
        )
    message = await context.bot.send_message(
        chat_id=chat_id,
        text="Генерируем ответ...",
    )
    answer = interact_manager(llm_model, question, context_data)
    await context.bot.delete_message(chat_id=chat_id, message_id=message.message_id)
    await context.bot.send_message(
            chat_id=chat_id,
            text=answer,
        )


def main():
    application = Application.builder().token(TG_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    application.run_polling()


if __name__ == "__main__":
    main()
