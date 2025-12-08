import wave
from pathlib import Path
from typing import Union

import pytest

from pysilero_vad import InvalidChunkSizeError, SileroVoiceActivityDetector

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
    assert all(p < 0.5 for p in vad.process_chunks(_load_wav(_DIR / "silence.wav")))


def test_speech() -> None:
    """Test VAD on recorded speech."""
    vad = SileroVoiceActivityDetector()
    assert any(p >= 0.5 for p in vad.process_chunks(_load_wav(_DIR / "speech.wav")))


def test_invalid_chunk_size() -> None:
    """Test that chunk size must be 512 samples."""
    vad = SileroVoiceActivityDetector()

    # Should work
    vad(bytes(SileroVoiceActivityDetector.chunk_bytes()))
    vad.process_samples(
        [0.0 for _ in range(SileroVoiceActivityDetector.chunk_samples())]
    )

    # Should fail
    with pytest.raises(InvalidChunkSizeError):
        vad(bytes(SileroVoiceActivityDetector.chunk_bytes() * 2))

    with pytest.raises(InvalidChunkSizeError):
        vad(bytes(SileroVoiceActivityDetector.chunk_bytes() // 2))

    with pytest.raises(InvalidChunkSizeError):
        vad.process_samples(
            [0.0 for _ in range(SileroVoiceActivityDetector.chunk_samples() * 2)]
        )

    with pytest.raises(InvalidChunkSizeError):
        vad.process_samples(
            [0.0 for _ in range(SileroVoiceActivityDetector.chunk_samples() // 2)]
        )
