import json
import queue as thread_queue
from datetime import datetime, timezone
from typing import Optional, Union

import requests
from fastapi import Body, Depends, FastAPI, HTTPException, Request

from config import (
    API_ACCESS_KEY,
    AI_CORE_URL,
    AI_CORE_TIMEOUT,
    CHAT_SESSIONS,
    LOGGER,
    RECENT_TRANSCRIPTS,
    TRANSCRIPTS_LOCK,
)
from models import ConversationRequest, ContextData, InputData, LegacyRequest

api_app = FastAPI(title="Telegram ASR Bot API")


def get_session_id(chat_id: int) -> str:
    if chat_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[chat_id] = f"tg-{chat_id}"
    return CHAT_SESSIONS[chat_id]


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


def call_ai_core(
    message: str, session_id: str, channel: str = "text", language_hint: str = "uz",
    images: Optional[list[str]] = None, agent: str = "",
) -> dict:
    payload = {
        "message": message,
        "session_id": session_id,
        "context": {
            "msisdn": None,
            "channel": channel,
            "language_hint": language_hint,
            "agent": agent,
        },
    }
    if images:
        payload["images"] = images
    LOGGER.info(
        "AI Core request: session=%s, channel=%s, msg=%s",
        session_id,
        channel,
        message[:100],
    )
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


_STREAM_DONE = None  # sentinel for end of stream


def call_ai_core_stream(
    message: str, session_id: str, channel: str = "text", language_hint: str = "uz",
    images: Optional[list[str]] = None, agent: str = "",
    chunk_queue: thread_queue.Queue | None = None,
) -> dict:
    """Call AI Core with streaming. Puts text chunks into chunk_queue as they arrive.
    Returns the full response dict. If AI Core doesn't support SSE, falls back
    to non-streaming and puts the entire text as a single chunk.
    Always puts _STREAM_DONE sentinel when finished (success or failure).
    """
    payload = {
        "message": message,
        "session_id": session_id,
        "stream": True,
        "context": {
            "msisdn": None,
            "channel": channel,
            "language_hint": language_hint,
            "agent": agent,
        },
    }
    if images:
        payload["images"] = images

    def _put(chunk: str | None):
        if chunk_queue is not None:
            chunk_queue.put(chunk)

    LOGGER.info(
        "AI Core stream request: session=%s, channel=%s, msg=%s",
        session_id, channel, message[:100],
    )

    try:
        response = requests.post(
            AI_CORE_URL, json=payload, timeout=AI_CORE_TIMEOUT,
            stream=True, headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        # --- Path A: AI Core supports SSE ---
        if "text/event-stream" in content_type:
            full_text = ""
            final_data = None
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        token = chunk.get("token", chunk.get("text", ""))
                        if token:
                            full_text += token
                            _put(token)
                        if chunk.get("done"):
                            final_data = chunk
                            break
                    except json.JSONDecodeError:
                        full_text += data_str
                        _put(data_str)
            _put(_STREAM_DONE)
            if final_data is None:
                final_data = {"response": {"text": full_text}}
            LOGGER.info("AI Core stream complete: %d chars", len(full_text))
            return final_data

        # --- Path B: AI Core returned plain JSON (no streaming support) ---
        data = response.json()
        text = data.get("response", {}).get("text", "")
        if text:
            _put(text)
        _put(_STREAM_DONE)
        LOGGER.info(
            "AI Core response (non-stream fallback): model=%s, latency=%sms",
            data.get("model_used", "?"), data.get("latency_ms", "?"),
        )
        return data

    except Exception:
        LOGGER.exception("AI Core stream request failed")
        _put(_STREAM_DONE)
        raise


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
    _: None = Depends(require_api_key),
) -> dict:
    # 1. Backward Compatibility Mapping
    if isinstance(request_data, LegacyRequest):
        LOGGER.warning(
            "Deprecated legacy request format received: {'message': ...}"
        )
        normalized_request = ConversationRequest(
            session_id=f"legacy-{datetime.now().timestamp()}",
            input=InputData(type="voice", content=request_data.message),
            context=ContextData(),
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

    # 3. Validation Logic
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
