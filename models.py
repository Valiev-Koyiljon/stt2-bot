from typing import Literal, Optional, Union

from pydantic import BaseModel


class ProcessingError(RuntimeError):
    pass


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
