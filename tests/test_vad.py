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


@pytest.mark.parametrize("num_chunks", [1, 2, 5])
def test_process_chunks_exact(num_chunks: int) -> None:
    """Audio of exactly N chunks gives N probabilities, including the last."""
    chunk_bytes = SileroVoiceActivityDetector.chunk_bytes()
    audio = _load_wav(_DIR / "speech.wav")[: chunk_bytes * num_chunks]
    assert len(audio) == chunk_bytes * num_chunks

    vad = SileroVoiceActivityDetector()
    probs = list(vad.process_chunks(audio))
    assert len(probs) == num_chunks

    # Same as processing each chunk in turn
    vad.reset()
    expected = [
        vad(audio[i : i + chunk_bytes]) for i in range(0, len(audio), chunk_bytes)
    ]
    assert probs == pytest.approx(expected)


@pytest.mark.parametrize("num_chunks", [1, 2, 5])
@pytest.mark.parametrize("extra_bytes", [2, 512, 1022])
def test_process_chunks_partial(num_chunks: int, extra_bytes: int) -> None:
    """A partial chunk at the end of the audio is ignored."""
    chunk_bytes = SileroVoiceActivityDetector.chunk_bytes()
    audio = bytes(chunk_bytes * num_chunks + extra_bytes)

    vad = SileroVoiceActivityDetector()
    assert len(list(vad.process_chunks(audio))) == num_chunks


def test_cpu_features() -> None:
    """A CPU backend is loaded, and says what it was built with."""
    import pysilero_vad  # pylint: disable=import-outside-toplevel

    assert isinstance(pysilero_vad.CPU_FEATURES, str)
