# Telegram ASR Bot

A Telegram bot that accepts voice/audio messages, converts them to 16 kHz mono WAV, splits audio into 30‑second chunks, sends each chunk to your ASR API, and replies with the combined transcription. It also exposes a small HTTP API for the most recent transcripts (no database).

## Features
- Accepts voice notes and audio files
- Converts to WAV (mono, 16 kHz)
- Splits into 30‑second chunks
- Calls ASR API with `X-API-Key` + `language`
- Replies with the final transcript
- Optional admin copy of every transcript
- HTTP API for recent transcripts
- Optional webhook callback per transcript

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

To get your chat ID, send `/id` to the bot. If you set `ADMIN_CHAT_ID` in `.env`,
the bot will send every transcript to you as well as the user.

## API
The API keeps a small in‑memory list of the most recent transcripts. Data is lost on restart.

Endpoints:
- `GET /health` -> `{ "status": "ok" }`
- `GET /last` -> `{ "item": { "username", "user_id", "text", "timestamp" } }`
- `GET /recent` -> `{ "items": [ ... ] }`

## Webhook (optional)
Set `WEBHOOK_URL` to receive a JSON payload for each transcript:
```json
{
  "username": "john",
  "user_id": 123456789,
  "text": "...",
  "timestamp": "2026-02-09T12:34:56+00:00"
}
```

## Docker Compose (Coolify)
This repo includes a `docker-compose.yml` that exposes port 7000 on the host to 8000 in the container:
- Host: `7000`
- Container: `8000`

In Coolify, set your environment variables and deploy the compose file.

## Configuration
All settings live in `.env`:
- `TELEGRAM_BOT_TOKEN`: your bot token
- `ASR_API_URL`: ASR endpoint (default set in `.env.example`)
- `ASR_API_KEY`: API key passed in `X-API-Key`
- `ASR_LANG`: language code (e.g., `uz`)
- `ASR_CHUNK_SECONDS`: chunk length (default 30)
- `FFMPEG_PATH`: path to `ffmpeg` if not in PATH
- `ADMIN_CHAT_ID`: your Telegram chat ID for receiving copies of transcripts
- `API_HOST`: API bind host (default `0.0.0.0`)
- `API_PORT`: API port inside container (default `8000`)
- `RECENT_LIMIT`: max number of transcripts in memory
- `WEBHOOK_URL`: optional webhook endpoint for transcript payloads
- `WEBHOOK_TIMEOUT`: webhook timeout in seconds

## Storage
Audio files are processed in a temporary folder and not saved. Transcripts are stored in memory only.
