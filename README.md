# Telegram ASR Bot

A Telegram bot that accepts voice/audio messages, converts them to 16 kHz mono WAV, splits audio into 30-second chunks, sends each chunk to your ASR API, and replies with the combined transcription. It also forwards the transcript to an AI Core Engine for intelligent responses. Exposes a small HTTP API for the most recent transcripts (no database).

## Features
- Accepts voice notes and audio files
- Converts to WAV (mono, 16 kHz)
- Splits into 30-second chunks
- **Parallel chunk transcription** (up to 3 concurrent ASR requests)
- Calls ASR API with `X-API-Key` header
- AI-powered responses via AI Core Engine
- Language selection: Uzbek / Russian (per-user preference)
- Optional admin copy of every transcript
- HTTP API for recent transcripts
- Optional webhook callback per transcript (fire-and-forget, non-blocking)

## Project Structure

```
stt-bot/
├── bot.py          # Entry point — registers handlers, starts API + polling
├── config.py       # Environment variables, logger, global state, constants
├── models.py       # Pydantic request models + ProcessingError
├── api.py          # FastAPI app, routes, AI Core integration
├── audio.py        # ffmpeg wrappers, ASR transcription
├── handlers.py     # Telegram command/message handlers + utilities
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## ASR API Latency Benchmarks

Tested against `185.100.53.247:8211/asr` (synthetic sine-wave audio, 3 runs each):

| Audio Duration | Avg Latency | Min    | Max    |
|---------------|-------------|--------|--------|
| 1 second      | ~130ms      | 123ms  | 135ms  |
| 5 seconds     | ~464ms      | 457ms  | 476ms  |
| 30 seconds    | ~936ms      | 904ms  | 1000ms |

> Benchmarked on 2026-02-13 from local machine against port 18000. Latency on port 8211 is expected to be similar. Real-world latency may vary with speech content, network conditions, and server load.
>
> With parallel transcription (up to 3 concurrent), a 2-minute audio (4 chunks) completes in ~1.9s instead of ~3.7s sequential.

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
The API keeps a small in-memory list of the most recent transcripts. Data is lost on restart.

Endpoints:
- `GET /health` -> `{ "status": "ok" }`
- `GET /last` -> `{ "item": { "session_id", "message_type", "content", "platform", "language", "username", "user_id", "timestamp" } }`
- `GET /recent` -> `{ "items": [ ... ] }`
- `POST /conversation` -> accepts `ConversationRequest` or legacy `{ "message": "..." }` format

If you set `API_ACCESS_KEY`, all API endpoints require the `X-API-Key` header.

## Webhook (optional)
Set `WEBHOOK_URL` to receive a JSON payload for each transcript:
```json
{
  "session_id": "tg-123456-1707500000.0",
  "message_type": "voice",
  "content": "transcribed text...",
  "platform": "telegram",
  "language": "auto",
  "username": "john",
  "user_id": 123456789,
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

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | (required) |
| `ASR_API_URL` | ASR endpoint | `http://185.100.53.247:8211/asr` |
| `ASR_API_KEY` | API key passed in `X-API-Key` | (empty) |
| `ASR_CHUNK_SECONDS` | Chunk length in seconds | `30` |
| `FFMPEG_PATH` | Path to `ffmpeg` binary | `ffmpeg` |
| `ADMIN_CHAT_ID` | Telegram chat ID for admin copies | (empty) |
| `AI_CORE_URL` | AI Core Engine endpoint | `http://185.100.53.247:8075/chat` |
| `AI_CORE_TIMEOUT` | AI Core request timeout (seconds) | `30` |
| `API_HOST` | API bind host | `0.0.0.0` |
| `API_PORT` | API port inside container | `8000` |
| `RECENT_LIMIT` | Max transcripts in memory | `100` |
| `API_ACCESS_KEY` | Protect API endpoints with `X-API-Key` | (empty) |
| `WEBHOOK_URL` | Webhook endpoint for transcript payloads | (empty) |
| `WEBHOOK_TIMEOUT` | Webhook timeout in seconds | `10` |

## Storage
Audio files are processed in a temporary folder and not saved. Transcripts are stored in memory only.
