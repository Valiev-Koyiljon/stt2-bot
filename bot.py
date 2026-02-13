import asyncio
import logging
import os
import subprocess
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from typing import Iterable, Union, Literal, Optional
from pydantic import BaseModel, Field

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Body
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

ASR_API_URL = os.getenv(
    "ASR_API_URL",
    "http://185.100.53.247:18000/v1/audio/transcriptions"
)
ASR_API_KEY = os.getenv("ASR_API_KEY", "")
ASR_LANG = os.getenv("ASR_LANG", "uz")
CHUNK_SECONDS = int(os.getenv("ASR_CHUNK_SECONDS", "30"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
RECENT_LIMIT = int(os.getenv("RECENT_LIMIT", "100"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "10"))
API_ACCESS_KEY = os.getenv("API_ACCESS_KEY", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("telegram-asr-bot")

api_app = FastAPI(title="Telegram ASR Bot API")
RECENT_TRANSCRIPTS: deque[dict] = deque(maxlen=RECENT_LIMIT)
TRANSCRIPTS_LOCK = threading.Lock()


class ProcessingError(RuntimeError):
    pass


# --- New Data Models ---

class InputData(BaseModel):
    type: Literal["text", "voice"]
    content: str

class ContextData(BaseModel):
    msisdn: Optional[str] = None
    platform: str = "telegram"
    language: str = "uz"
    username: Optional[str] = None
    user_id: Optional[Union[int, str]] = None

class ConversationRequest(BaseModel):
    session_id: str
    input: InputData
    context: ContextData

class LegacyRequest(BaseModel):
    message: str


# --- API Logic ---


def require_api_key(request: Request) -> None:
    if not API_ACCESS_KEY:
        return
    header_key = request.headers.get("X-API-Key", "")
    if header_key != API_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@api_app.get("/health")
def health(_: None = Depends(require_api_key)) -> dict:
    return {"status": "ok"}


@api_app.get("/last")
def last_transcript(_: None = Depends(require_api_key)) -> dict:
    with TRANSCRIPTS_LOCK:
        if not RECENT_TRANSCRIPTS:
            return {"item": None}
        return {"item": RECENT_TRANSCRIPTS[-1]}


@api_app.get("/recent")
def recent_transcripts(_: None = Depends(require_api_key)) -> dict:
    with TRANSCRIPTS_LOCK:
        return {"items": list(RECENT_TRANSCRIPTS)}


def store_message(
    session_id: str,
    message_type: str,
    content: str,
    platform: str,
    language: str,
    msisdn: Optional[str] = None,
    username: Optional[str] = None,
    user_id: Optional[Union[int, str]] = None,
) -> dict:
    payload = {
        "session_id": session_id,
        "message_type": message_type,
        "content": content,
        "platform": platform,
        "language": language,
        "msisdn": msisdn,
        "username": username,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with TRANSCRIPTS_LOCK:
        RECENT_TRANSCRIPTS.append(payload)
    return payload


def process_business_logic(content: str) -> str:
    # Placeholder for actual business logic
    # In a real scenario, this might call another service or AI model
    return f"Processed: {content}"


@api_app.post("/conversation")
def handle_conversation(
    request_data: Union[ConversationRequest, LegacyRequest] = Body(...),
    _: None = Depends(require_api_key)
) -> dict:
    # 1. Backward Compatibility Mapping
    if isinstance(request_data, LegacyRequest):
        LOGGER.warning("Deprecated legacy request format received: {'message': ...}")
        normalized_request = ConversationRequest(
            session_id=f"legacy-{datetime.now().timestamp()}",
            input=InputData(type="voice", content=request_data.message),
            context=ContextData()
        )
    else:
        normalized_request = request_data

    # 2. Extract Data
    session_id = normalized_request.session_id
    msg_type = normalized_request.input.type
    content = normalized_request.input.content
    platform = normalized_request.context.platform
    language = normalized_request.context.language
    msisdn = normalized_request.context.msisdn

    # 3. Validation Logic (Already handled by Pydantic, but can add more)
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    # 4. Store Message
    store_message(
        session_id=session_id,
        message_type=msg_type,
        content=content,
        platform=platform,
        language=language,
        msisdn=msisdn,
        username=normalized_request.context.username,
        user_id=normalized_request.context.user_id,
    )

    # 5. Process Business Logic
    reply = process_business_logic(content)

    # 6. Return Response
    return {
        "session_id": session_id,
        "reply": reply
    }


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        tail = stderr[-1000:] if stderr else "(no stderr)"
        raise ProcessingError(f"ffmpeg failed: {tail}")


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    run_ffmpeg(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
        ]
    )


def split_wav(input_path: Path, output_dir: Path, chunk_seconds: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "chunk_%03d.wav"
    run_ffmpeg(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            str(input_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c",
            "pcm_s16le",
            str(output_pattern),
        ]
    )
    return sorted(output_dir.glob("chunk_*.wav"))


def extract_text(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("text", "result", "transcript", "transcription"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(payload).strip()


def transcribe_chunk(chunk_path: Path) -> str:
    headers = {"X-API-Key": ASR_API_KEY} if ASR_API_KEY else {}
    with chunk_path.open("rb") as audio_file:
        response = requests.post(
            ASR_API_URL,
            files={"audio": audio_file},
            data={"language": ASR_LANG},
            headers=headers,
            timeout=60,
        )
    response.raise_for_status()
    return extract_text(response.json())


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
    # Use the unified store_message for consistency
    return store_message(
        session_id=f"tg-{message.chat_id}-{datetime.now().timestamp()}",
        message_type="voice",  # Bot handles audio/voice
        content=text,
        platform="telegram",
        language=ASR_LANG,
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


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    # Store text message using the unified schema
    payload = store_message(
        session_id=f"tg-{message.chat_id}-{datetime.now().timestamp()}",
        message_type="text",
        content=message.text,
        platform="telegram",
        language=ASR_LANG,  # Default or infer if possible
        msisdn=None,
        username=message.from_user.username,
        user_id=message.from_user.id,
    )

    # Process business logic (reply)
    reply = process_business_logic(message.text)
    
    # Send reply back to user
    await message.reply_text(reply)

    # Webhook
    await asyncio.to_thread(post_webhook, payload)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a voice or audio file. I will split it into 30-second WAV chunks and transcribe it."
    )


async def show_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(f"Your chat ID: {update.message.chat_id}")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    attachment = pick_attachment(update)
    if not attachment:
        await message.reply_text("Please send a voice note or audio file.")
        return

    if not TELEGRAM_BOT_TOKEN:
        await message.reply_text("Bot token is not configured. Set TELEGRAM_BOT_TOKEN.")
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
            chunk_paths = await asyncio.to_thread(split_wav, wav_path, chunks_dir, CHUNK_SECONDS)
            if not chunk_paths:
                raise ProcessingError("No audio chunks were created.")

            transcripts: list[str] = []
            total = len(chunk_paths)
            for idx, chunk_path in enumerate(chunk_paths, start=1):
                await status_message.edit_text(f"Transcribing chunk {idx}/{total}...")
                text = await asyncio.to_thread(transcribe_chunk, chunk_path)
                if text:
                    transcripts.append(text)

            full_text = " ".join(t.strip() for t in transcripts if t).strip()
            if not full_text:
                full_text = "No text returned by ASR service."

            payload = record_transcript(message, full_text)
            await asyncio.to_thread(post_webhook, payload)

            for part in split_message(full_text):
                await message.reply_text(part)
                if ADMIN_CHAT_ID and str(message.chat_id) != ADMIN_CHAT_ID:
                    await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=part)

    except requests.HTTPError as exc:
        LOGGER.exception("ASR API error")
        await message.reply_text(f"ASR API error: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Processing error")
        await message.reply_text(f"Processing failed: {exc}")
    finally:
        try:
            await status_message.delete()
        except Exception:  # pylint: disable=broad-except
            pass


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
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
