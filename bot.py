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
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# ASR_API_URL = os.getenv("ASR_API_URL", "http://185.100.53.247:7190/asr")
ASR_API_URL = os.getenv(
    "ASR_API_URL",
    "http://185.100.53.247:18000/asr"
)
ASR_API_KEY = os.getenv("ASR_API_KEY", "")
CHUNK_SECONDS = int(os.getenv("ASR_CHUNK_SECONDS", "30"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()
AI_CORE_URL = os.getenv("AI_CORE_URL", "http://185.100.53.247:8075/chat")
AI_CORE_TIMEOUT = int(os.getenv("AI_CORE_TIMEOUT", "30"))

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
CHAT_SESSIONS: dict[int, str] = {}
USER_LANGUAGES: dict[int, str] = {}  # chat_id -> "uz" or "ru"

# Persistent reply keyboard buttons
BTN_UZ = "O'zbek \U0001f1fa\U0001f1ff"
BTN_RU = "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \U0001f1f7\U0001f1fa"
LANG_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(BTN_UZ), KeyboardButton(BTN_RU)]],
    resize_keyboard=True,
)
LANG_BUTTON_MAP = {BTN_UZ: "uz", BTN_RU: "ru"}


class ProcessingError(RuntimeError):
    pass


def get_session_id(chat_id: int) -> str:
    if chat_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[chat_id] = f"tg-{chat_id}"
    return CHAT_SESSIONS[chat_id]


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


def call_ai_core(message: str, session_id: str, channel: str = "text", language_hint: str = "uz") -> dict:
    payload = {
        "message": message,
        "session_id": session_id,
        "context": {
            "msisdn": None,
            "channel": channel,
            "language_hint": language_hint,
        },
    }
    LOGGER.info("AI Core request: session=%s, channel=%s, msg=%s", session_id, channel, message[:100])
    response = requests.post(AI_CORE_URL, json=payload, timeout=AI_CORE_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    LOGGER.info(
        "AI Core response: model=%s, latency=%sms, tools=%s",
        data.get("model_used", "?"),
        data.get("latency_ms", "?"),
        data.get("tool_calls_made", []),
    )
    return data


def format_ai_response(ai_data: dict) -> str:
    parts = []
    response_obj = ai_data.get("response", {})
    text = response_obj.get("text", "").strip()
    if text:
        parts.append(text)
    tool_calls = ai_data.get("tool_calls_made", [])
    if tool_calls:
        parts.append(f"\n\U0001f527 Tools: {', '.join(tool_calls)}")
    latency_ms = ai_data.get("latency_ms")
    model_used = ai_data.get("model_used", "")
    if latency_ms is not None:
        parts.append(f"\u23f1 {latency_ms}ms | {model_used}")
    return "\n".join(parts) if parts else "No response from AI."


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

    # 5. Call AI Core Engine
    ai_core_response = None
    try:
        ai_data = call_ai_core(
            message=content,
            session_id=session_id,
            channel="voice" if msg_type == "voice" else "text",
            language_hint=language,
        )
        reply = ai_data.get("response", {}).get("text", "")
        ai_core_response = ai_data
    except Exception as exc:
        LOGGER.exception("AI Core Engine call failed in /conversation")
        reply = f"AI service error: {exc}"

    # 6. Return Response
    return {
        "session_id": session_id,
        "reply": reply,
        "ai_core_response": ai_core_response,
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


def transcribe_chunk(chunk_path: Path, language: str = "") -> str:
    headers = {"X-API-Key": ASR_API_KEY} if ASR_API_KEY else {}
    data = {"language": language} if language else {}
    with chunk_path.open("rb") as audio_file:
        response = requests.post(
            ASR_API_URL,
            files={"audio": audio_file},
            data=data,
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
            call_ai_core, message.text, session_id, channel="text", language_hint=user_lang,
        )

        tool_calls = ai_data.get("tool_calls_made", [])
        if tool_calls:
            await status_message.edit_text(f"\U0001f527 Using tools: {', '.join(tool_calls)}")
            await asyncio.sleep(0.5)

        reply_text = format_ai_response(ai_data)
        for part in split_message(reply_text):
            await message.reply_text(part)
            if ADMIN_CHAT_ID and str(message.chat_id) != ADMIN_CHAT_ID:
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=part)

    except requests.ConnectionError:
        LOGGER.exception("AI Core Engine connection failed")
        await message.reply_text("\u26a0\ufe0f AI service is currently unavailable. Please try again later.")
    except requests.Timeout:
        LOGGER.exception("AI Core Engine timed out")
        await message.reply_text("\u26a0\ufe0f AI service timed out. Please try again.")
    except requests.HTTPError as exc:
        LOGGER.exception("AI Core Engine HTTP error")
        await message.reply_text(f"\u26a0\ufe0f AI service error: {exc.response.status_code}")
    except Exception as exc:
        LOGGER.exception("Error calling AI Core Engine")
        await message.reply_text(f"\u26a0\ufe0f Processing failed: {exc}")
    finally:
        try:
            await status_message.delete()
        except Exception:
            pass

    await asyncio.to_thread(post_webhook, payload)


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
                f"{'> ' if current == 'uz' else ''}O'zbek", callback_data="lang_uz"
            ),
            InlineKeyboardButton(
                f"{'> ' if current == 'ru' else ''}Русский", callback_data="lang_ru"
            ),
        ]
    ]
    label = "O\u2019zbek" if current == "uz" else "Русский"
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
    label = "O\u2019zbek" if lang_code == "uz" else "Русский"
    await query.edit_message_text(f"Language set to: {label}")


async def handle_lang_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    lang_code = LANG_BUTTON_MAP.get(text)
    if not lang_code:
        return
    chat_id = update.message.chat_id
    USER_LANGUAGES[chat_id] = lang_code
    label = "O\u2019zbek \U0001f1fa\U0001f1ff" if lang_code == "uz" else "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \U0001f1f7\U0001f1fa"
    await update.message.reply_text(f"\u2705 Language set to: {label}")


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

            user_lang = USER_LANGUAGES.get(message.chat_id, "")
            transcripts: list[str] = []
            total = len(chunk_paths)
            for idx, chunk_path in enumerate(chunk_paths, start=1):
                await status_message.edit_text(f"Transcribing chunk {idx}/{total}...")
                text = await asyncio.to_thread(transcribe_chunk, chunk_path, user_lang)
                if text:
                    transcripts.append(text)

            full_text = " ".join(t.strip() for t in transcripts if t).strip()
            if not full_text:
                full_text = "No text returned by ASR service."

            payload = record_transcript(message, full_text)
            await asyncio.to_thread(post_webhook, payload)

            for part in split_message(f"\U0001f4dd Transcript:\n{full_text}"):
                await message.reply_text(part)

            # Call AI Core Engine with the transcribed text
            session_id = get_session_id(message.chat_id)
            await status_message.edit_text("\U0001f914 Thinking...")

            try:
                ai_data = await asyncio.to_thread(
                    call_ai_core, full_text, session_id, channel="voice", language_hint=user_lang or "uz",
                )

                tool_calls = ai_data.get("tool_calls_made", [])
                if tool_calls:
                    await status_message.edit_text(f"\U0001f527 Using tools: {', '.join(tool_calls)}")
                    await asyncio.sleep(0.5)

                reply_text = format_ai_response(ai_data)
                for part in split_message(reply_text):
                    await message.reply_text(part)
                    if ADMIN_CHAT_ID and str(message.chat_id) != ADMIN_CHAT_ID:
                        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=part)

            except requests.ConnectionError:
                LOGGER.exception("AI Core Engine connection failed")
                await message.reply_text("\u26a0\ufe0f AI service unavailable. Transcript was saved.")
            except requests.Timeout:
                LOGGER.exception("AI Core Engine timed out")
                await message.reply_text("\u26a0\ufe0f AI service timed out. Transcript was saved.")
            except Exception as exc:
                LOGGER.exception("AI Core processing failed")
                await message.reply_text(f"\u26a0\ufe0f AI processing failed: {exc}. Transcript was saved.")

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
    app.add_handler(CommandHandler("lang", lang_command))
    app.add_handler(CallbackQueryHandler(lang_callback, pattern="^lang_"))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))
    lang_btn_filter = filters.TEXT & filters.Regex(f"^({BTN_UZ}|{BTN_RU})$")
    app.add_handler(MessageHandler(lang_btn_filter, handle_lang_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
