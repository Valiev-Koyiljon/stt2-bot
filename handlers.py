import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from config import (
    ADMIN_CHAT_ID,
    CHUNK_SECONDS,
    LOGGER,
    TELEGRAM_BOT_TOKEN,
    USER_LANGUAGES,
    WEBHOOK_URL,
    WEBHOOK_TIMEOUT,
    BTN_UZ,
    BTN_RU,
    LANG_KEYBOARD,
    LANG_BUTTON_MAP,
)
from models import ProcessingError
from api import call_ai_core, format_ai_response, get_session_id, store_message
from audio import convert_to_wav, split_wav, transcribe_chunk


# --- Utilities ---


def split_message(text: str, limit: int = 4000) -> Iterable[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts = []
    current = []
    current_len = 0
    for word in text.split():
        if current_len + len(word) + 1 > limit:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def pick_attachment(update: Update):
    message = update.message
    if not message:
        return None
    if message.voice:
        return message.voice
    if message.audio:
        return message.audio
    if message.document and message.document.mime_type:
        if message.document.mime_type.startswith("audio/"):
            return message.document
    return None


def record_transcript(message, text: str) -> dict:
    user = message.from_user
    return store_message(
        session_id=f"tg-{message.chat_id}-{datetime.now().timestamp()}",
        message_type="voice",
        content=text,
        platform="telegram",
        language="auto",
        msisdn=None,
        username=user.username,
        user_id=user.id,
    )


def post_webhook(payload: dict) -> None:
    if not WEBHOOK_URL:
        return
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=WEBHOOK_TIMEOUT)
        response.raise_for_status()
    except Exception:
        LOGGER.exception("Webhook delivery failed")


# --- Telegram Handlers ---


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a voice or text message. I will transcribe voice and respond using AI.\n\n"
        "Use the buttons below to switch STT language.\n\n"
        "Commands:\n"
        "/start - Show this help\n"
        "/id - Show your chat ID",
        reply_markup=LANG_KEYBOARD,
    )


async def show_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(f"Your chat ID: {update.message.chat_id}")


async def lang_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    current = USER_LANGUAGES.get(chat_id, "uz")
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'> ' if current == 'uz' else ''}O'zbek",
                callback_data="lang_uz",
            ),
            InlineKeyboardButton(
                f"{'> ' if current == 'ru' else ''}\u0420\u0443\u0441\u0441\u043a\u0438\u0439",
                callback_data="lang_ru",
            ),
        ]
    ]
    label = "O\u2019zbek" if current == "uz" else "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"
    await update.message.reply_text(
        f"Current language: {label}\nChoose STT language:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    lang_code = query.data.replace("lang_", "")  # "uz" or "ru"
    chat_id = query.message.chat_id
    USER_LANGUAGES[chat_id] = lang_code
    label = "O\u2019zbek" if lang_code == "uz" else "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"
    await query.edit_message_text(f"Language set to: {label}")


async def handle_lang_button(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    text = update.message.text
    lang_code = LANG_BUTTON_MAP.get(text)
    if not lang_code:
        return
    chat_id = update.message.chat_id
    USER_LANGUAGES[chat_id] = lang_code
    label = (
        "O\u2019zbek \U0001f1fa\U0001f1ff"
        if lang_code == "uz"
        else "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \U0001f1f7\U0001f1fa"
    )
    await update.message.reply_text(f"\u2705 Language set to: {label}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    session_id = get_session_id(message.chat_id)

    payload = store_message(
        session_id=session_id,
        message_type="text",
        content=message.text,
        platform="telegram",
        language="auto",
        msisdn=None,
        username=message.from_user.username,
        user_id=message.from_user.id,
    )

    await message.chat.send_action(ChatAction.TYPING)
    status_message = await message.reply_text("\U0001f914 Thinking...")

    try:
        user_lang = USER_LANGUAGES.get(message.chat_id, "uz")
        ai_data = await asyncio.to_thread(
            call_ai_core,
            message.text,
            session_id,
            channel="text",
            language_hint=user_lang,
        )

        tool_calls = ai_data.get("tool_calls_made", [])
        if tool_calls:
            await status_message.edit_text(
                f"\U0001f527 Using tools: {', '.join(tool_calls)}"
            )
            await asyncio.sleep(0.5)

        reply_text = format_ai_response(ai_data)
        for part in split_message(reply_text):
            await message.reply_text(part)
            if ADMIN_CHAT_ID and str(message.chat_id) != ADMIN_CHAT_ID:
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=part)

    except requests.ConnectionError:
        LOGGER.exception("AI Core Engine connection failed")
        await message.reply_text(
            "\u26a0\ufe0f AI service is currently unavailable. Please try again later."
        )
    except requests.Timeout:
        LOGGER.exception("AI Core Engine timed out")
        await message.reply_text(
            "\u26a0\ufe0f AI service timed out. Please try again."
        )
    except requests.HTTPError as exc:
        LOGGER.exception("AI Core Engine HTTP error")
        await message.reply_text(
            f"\u26a0\ufe0f AI service error: {exc.response.status_code}"
        )
    except Exception as exc:
        LOGGER.exception("Error calling AI Core Engine")
        await message.reply_text(f"\u26a0\ufe0f Processing failed: {exc}")
    finally:
        try:
            await status_message.delete()
        except Exception:
            pass

    asyncio.create_task(asyncio.to_thread(post_webhook, payload))


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    attachment = pick_attachment(update)
    if not attachment:
        await message.reply_text("Please send a voice note or audio file.")
        return

    if not TELEGRAM_BOT_TOKEN:
        await message.reply_text(
            "Bot token is not configured. Set TELEGRAM_BOT_TOKEN."
        )
        return

    await message.chat.send_action(ChatAction.TYPING)
    status_message = await message.reply_text("Downloading audio...")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tg_file = await attachment.get_file()
            suffix = Path(tg_file.file_path or "audio").suffix
            input_path = tmp_path / f"input{suffix or '.dat'}"
            wav_path = tmp_path / "converted.wav"
            chunks_dir = tmp_path / "chunks"

            await tg_file.download_to_drive(custom_path=str(input_path))
            await status_message.edit_text("Converting to WAV...")
            await asyncio.to_thread(convert_to_wav, input_path, wav_path)

            await status_message.edit_text("Splitting into chunks...")
            chunk_paths = await asyncio.to_thread(
                split_wav, wav_path, chunks_dir, CHUNK_SECONDS
            )
            if not chunk_paths:
                raise ProcessingError("No audio chunks were created.")

            user_lang = USER_LANGUAGES.get(message.chat_id, "")
            total = len(chunk_paths)
            await status_message.edit_text(
                f"Transcribing {total} chunk{'s' if total > 1 else ''}..."
            )

            sem = asyncio.Semaphore(3)

            async def _transcribe(chunk_path: Path) -> str:
                async with sem:
                    return await asyncio.to_thread(
                        transcribe_chunk, chunk_path, user_lang
                    )

            results = await asyncio.gather(
                *[_transcribe(cp) for cp in chunk_paths]
            )
            transcripts = [t for t in results if t]

            full_text = " ".join(t.strip() for t in transcripts if t).strip()
            if not full_text:
                full_text = "No text returned by ASR service."

            payload = record_transcript(message, full_text)
            asyncio.create_task(asyncio.to_thread(post_webhook, payload))

            for part in split_message(f"\U0001f4dd Transcript:\n{full_text}"):
                await message.reply_text(part)

            # Call AI Core Engine with the transcribed text
            session_id = get_session_id(message.chat_id)
            await status_message.edit_text("\U0001f914 Thinking...")

            try:
                ai_data = await asyncio.to_thread(
                    call_ai_core,
                    full_text,
                    session_id,
                    channel="voice",
                    language_hint=user_lang or "uz",
                )

                tool_calls = ai_data.get("tool_calls_made", [])
                if tool_calls:
                    await status_message.edit_text(
                        f"\U0001f527 Using tools: {', '.join(tool_calls)}"
                    )
                    await asyncio.sleep(0.5)

                reply_text = format_ai_response(ai_data)
                for part in split_message(reply_text):
                    await message.reply_text(part)
                    if ADMIN_CHAT_ID and str(message.chat_id) != ADMIN_CHAT_ID:
                        await context.bot.send_message(
                            chat_id=ADMIN_CHAT_ID, text=part
                        )

            except requests.ConnectionError:
                LOGGER.exception("AI Core Engine connection failed")
                await message.reply_text(
                    "\u26a0\ufe0f AI service unavailable. Transcript was saved."
                )
            except requests.Timeout:
                LOGGER.exception("AI Core Engine timed out")
                await message.reply_text(
                    "\u26a0\ufe0f AI service timed out. Transcript was saved."
                )
            except Exception as exc:
                LOGGER.exception("AI Core processing failed")
                await message.reply_text(
                    f"\u26a0\ufe0f AI processing failed: {exc}. Transcript was saved."
                )

    except requests.HTTPError as exc:
        LOGGER.exception("ASR API error")
        await message.reply_text(f"ASR API error: {exc}")
    except Exception as exc:
        LOGGER.exception("Processing error")
        await message.reply_text(f"Processing failed: {exc}")
    finally:
        try:
            await status_message.delete()
        except Exception:
            pass
