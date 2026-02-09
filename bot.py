import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ASR_API_URL = os.getenv("ASR_API_URL", "http://185.100.53.247:7190/asr")
ASR_API_KEY = os.getenv("ASR_API_KEY", "aifirstm")
ASR_LANG = os.getenv("ASR_LANG", "uz")
CHUNK_SECONDS = int(os.getenv("ASR_CHUNK_SECONDS", "30"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("telegram-asr-bot")


class ProcessingError(RuntimeError):
    pass


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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a voice or audio file. I will split it into 30-second WAV chunks and transcribe it."
    )


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
            wav_path = tmp_path / "input.wav"
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

            for part in split_message(full_text):
                await message.reply_text(part)

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
    return (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )


def main() -> None:
    app = build_application()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
