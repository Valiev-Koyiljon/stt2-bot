import asyncio
import base64
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
    AGENTS,
    CHAT_AGENTS,
    CHUNK_SECONDS,
    DEFAULT_AGENT,
    LOGGER,
    STORAGE_API_KEY,
    STORAGE_API_URL,
    TELEGRAM_BOT_TOKEN,
    WEBHOOK_URL,
    WEBHOOK_TIMEOUT,
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


def upload_to_storage(
    audio_data: bytes,
    filename: str,
    transcript: str,
    user_id: str,
    username: str,
    chat_id: str,
    message_id: str,
    session_id: str,
) -> None:
    if not STORAGE_API_URL:
        return
    try:
        resp = requests.post(
            f"{STORAGE_API_URL}/upload",
            files={"audio": (filename, audio_data)},
            data={
                "transcript": transcript,
                "user_id": str(user_id),
                "username": username or "",
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "session_id": session_id,
            },
            headers={"X-API-Key": STORAGE_API_KEY} if STORAGE_API_KEY else {},
            timeout=30,
        )
        resp.raise_for_status()
        LOGGER.info("Voice uploaded to storage: %s", resp.json().get("path"))
    except Exception:
        LOGGER.exception("Storage upload failed")


# --- Telegram Handlers ---


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome! I'm an AI-powered voice assistant.\n\n"
        "What I can do:\n"
        "\u2022 Voice/audio messages \u2014 I transcribe your speech and respond with AI\n"
        "\u2022 Text messages \u2014 I respond directly with AI\n\n"
        "Language is detected automatically \u2014 just speak naturally.\n\n"
        "/agent \u2014 Choose an AI agent\n"
        "/id \u2014 Show your chat ID",
    )


async def show_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(f"Your chat ID: {update.message.chat_id}")


def _get_agent(chat_id: int) -> str:
    return CHAT_AGENTS.get(chat_id, DEFAULT_AGENT)


async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    current = _get_agent(update.message.chat_id)
    buttons = []
    for agent_id, agent_name in AGENTS:
        label = f"{'> ' if agent_id == current else ''}{agent_name}"
        buttons.append(
            [InlineKeyboardButton(label, callback_data=f"agent:{agent_id}")]
        )
    await update.message.reply_text(
        "Select an agent:", reply_markup=InlineKeyboardMarkup(buttons)
    )


async def agent_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("agent:"):
        return
    await query.answer()
    agent_id = query.data.split(":", 1)[1]
    agent_name = next((n for a, n in AGENTS if a == agent_id), agent_id)
    CHAT_AGENTS[query.message.chat_id] = agent_id
    LOGGER.info("Chat %s switched to agent: %s", query.message.chat_id, agent_id)
    await query.edit_message_text(f"Agent switched to: {agent_name}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.photo:
        return

    session_id = get_session_id(message.chat_id)
    text = message.caption or "Describe this image"

    await message.chat.send_action(ChatAction.TYPING)
    status_message = await message.reply_text("\U0001f4f7 Processing image...")

    try:
        # Download highest-resolution photo
        photo = message.photo[-1]
        tg_file = await photo.get_file()
        photo_bytes = await tg_file.download_as_bytearray()
        image_b64 = base64.b64encode(bytes(photo_bytes)).decode("utf-8")

        await status_message.edit_text("\U0001f914 Thinking...")

        ai_data = await asyncio.to_thread(
            call_ai_core,
            text,
            session_id,
            channel="text",
            language_hint="auto",
            images=[image_b64],
            agent=_get_agent(message.chat_id),
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
        LOGGER.exception("Error processing photo")
        await message.reply_text(f"\u26a0\ufe0f Processing failed: {exc}")
    finally:
        try:
            await status_message.delete()
        except Exception:
            pass


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
        ai_data = await asyncio.to_thread(
            call_ai_core,
            message.text,
            session_id,
            channel="text",
            language_hint="auto",
            agent=_get_agent(message.chat_id),
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

            total = len(chunk_paths)
            await status_message.edit_text(
                f"Transcribing {total} chunk{'s' if total > 1 else ''}..."
            )

            sem = asyncio.Semaphore(3)

            async def _transcribe(chunk_path: Path) -> str:
                async with sem:
                    return await asyncio.to_thread(
                        transcribe_chunk, chunk_path
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

            session_id = get_session_id(message.chat_id)
            audio_data = input_path.read_bytes()
            asyncio.create_task(
                asyncio.to_thread(
                    upload_to_storage,
                    audio_data,
                    input_path.name,
                    full_text,
                    str(message.from_user.id),
                    message.from_user.username or "",
                    str(message.chat_id),
                    str(message.message_id),
                    session_id,
                )
            )

            for part in split_message(f"\U0001f4dd Transcript:\n{full_text}"):
                await message.reply_text(part)

            # Call AI Core Engine with the transcribed text
            await status_message.edit_text("\U0001f914 Thinking...")

            try:
                ai_data = await asyncio.to_thread(
                    call_ai_core,
                    full_text,
                    session_id,
                    channel="voice",
                    language_hint="auto",
                    agent=_get_agent(message.chat_id),
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
