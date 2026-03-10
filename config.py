import logging
import os
import threading
from collections import deque

from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ASR_API_URL = os.getenv(
    "ASR_API_URL",
    "http://185.100.53.247:8211/asr",
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

STORAGE_API_URL = os.getenv("STORAGE_API_URL", "").strip()
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "").strip()

# --- Agents ---
def _parse_agents(raw: str) -> list[tuple[str, str]]:
    """Parse 'id:Name,id:Name' into [(id, name), ...]."""
    agents = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            agent_id, name = entry.split(":", 1)
            agents.append((agent_id.strip(), name.strip()))
    return agents

_agents_raw = os.getenv("AGENTS", "default:Default").strip()
AGENTS: list[tuple[str, str]] = _parse_agents(_agents_raw)
DEFAULT_AGENT: str = AGENTS[0][0] if AGENTS else "default"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("telegram-asr-bot")

# --- Global Mutable State ---
RECENT_TRANSCRIPTS: deque[dict] = deque(maxlen=RECENT_LIMIT)
TRANSCRIPTS_LOCK = threading.Lock()
CHAT_SESSIONS: dict[int, str] = {}
CHAT_AGENTS: dict[int, str] = {}
AUTHENTICATED_CHATS: set[int] = set()

BOT_PASSWORD = os.getenv("BOT_PASSWORD", "").strip()
