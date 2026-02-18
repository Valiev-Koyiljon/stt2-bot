import logging
import os
import threading
from collections import deque

from dotenv import load_dotenv
from telegram import KeyboardButton, ReplyKeyboardMarkup

load_dotenv()

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ASR_API_URL = os.getenv(
    "ASR_API_URL",
    "http://185.100.53.247:18000/asr",
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
USER_LANGUAGES: dict[int, str] = {}  # chat_id -> "uz" or "ru"

# --- Language Keyboard ---
BTN_UZ = "O'zbek \U0001f1fa\U0001f1ff"
BTN_RU = "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \U0001f1f7\U0001f1fa"
LANG_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(BTN_UZ), KeyboardButton(BTN_RU)]],
    resize_keyboard=True,
)
LANG_BUTTON_MAP = {BTN_UZ: "uz", BTN_RU: "ru"}
