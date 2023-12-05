import wave
from pathlib import Path
from typing import Union

from pysilero_vad import SileroVoiceActivityDetector

_DIR = Path(__file__).parent


def _load_wav(wav_path: Union[str, Path]) -> bytes:
    """Return audio bytes from a WAV file."""
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        return wav_file.readframes(wav_file.getnframes())


def test_silence() -> None:
    """Test VAD on recorded silence."""
    vad = SileroVoiceActivityDetector()
    assert vad(_load_wav(_DIR / "silence.wav")) < 0.5


def test_speech() -> None:
    """Test VAD on recorded speech."""
    vad = SileroVoiceActivityDetector()
    assert vad(_load_wav(_DIR / "speech.wav")) >= 0.5
