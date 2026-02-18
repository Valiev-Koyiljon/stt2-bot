import subprocess
from pathlib import Path

import requests

from config import ASR_API_KEY, ASR_API_URL, FFMPEG_PATH
from models import ProcessingError


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


def transcribe_chunk(chunk_path: Path, language: str = "auto") -> str:
    headers = {"X-API-Key": ASR_API_KEY} if ASR_API_KEY else {}
    data = {"language": language}
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
