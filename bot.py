"""Telegram ASR Bot - entry point."""

import threading

import uvicorn
from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import API_HOST, API_PORT, LOGGER, TELEGRAM_BOT_TOKEN
from api import api_app
from handlers import (
    agent_callback,
    agent_command,
    handle_audio,
    handle_photo,
    handle_text,
    show_id,
    start,
)


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    return Application.builder().token(TELEGRAM_BOT_TOKEN).build()


def start_api_thread() -> None:
    thread = threading.Thread(
        target=uvicorn.run,
        args=(api_app,),
        kwargs={"host": API_HOST, "port": API_PORT, "log_level": "info"},
        daemon=True,
    )
    thread.start()
    LOGGER.info("API server running on http://%s:%s", API_HOST, API_PORT)


def main() -> None:
    start_api_thread()
    app = build_application()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("id", show_id))
    app.add_handler(CommandHandler("agent", agent_command))
    app.add_handler(CallbackQueryHandler(agent_callback, pattern=r"^agent:"))
    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio
        )
    )
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
