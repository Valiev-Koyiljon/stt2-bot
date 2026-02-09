# Telegram ASR Bot

A Telegram bot that accepts voice/audio messages, converts them to 16 kHz mono WAV, splits audio into 30‑second chunks, sends each chunk to your ASR API, and replies with the combined transcription.

## Features
- Accepts voice notes and audio files
- Converts to WAV (mono, 16 kHz)
- Splits into 30‑second chunks
- Calls ASR API with `X-API-Key` + `language`
- Replies with the final transcript

## Requirements
- Python
- `ffmpeg` available in PATH (or set `FFMPEG_PATH`)

## Install ffmpeg
macOS (Homebrew):
```bash
brew install ffmpeg
```

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Windows:
1. Download from https://ffmpeg.org/download.html
2. Extract and add the `bin` folder to your PATH

## Setup
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file (copy from `.env.example`) and fill values:
   ```bash
   cp .env.example .env
   ```

## Get a Telegram Bot Token
1. Open Telegram and start a chat with `@BotFather`.
2. Send `/newbot`.
3. Choose a name and a username (must end with `bot`).
4. Copy the token you receive and put it into `.env`:
   ```env
   TELEGRAM_BOT_TOKEN=YOUR_TOKEN_HERE
   ```

## Run
```bash
python bot.py
```

## Usage
Send a voice note or audio file to the bot. It will reply with the transcription.

## Configuration
All settings live in `.env`:
- `TELEGRAM_BOT_TOKEN`: your bot token
- `ASR_API_URL`: ASR endpoint (default set in `.env.example`)
- `ASR_API_KEY`: API key passed in `X-API-Key`
- `ASR_LANG`: language code (e.g., `uz`)
- `ASR_CHUNK_SECONDS`: chunk length (default 30)
- `FFMPEG_PATH`: path to `ffmpeg` if not in PATH

## Storage
Audio files are processed in a temporary folder and not saved. Transcripts are sent back to the chat only. If you want to persist audio/transcripts to disk or a channel, we can add that.
