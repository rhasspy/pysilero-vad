import logging
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Iterable, Union

# pylint: disable=no-name-in-module
from .silero_vad import load_model as _load_model
from .silero_vad import process_chunk as _process_chunk
from .silero_vad import reset as _reset

_MAX_WAV: Final = 32767
_DIR = Path(__file__).parent
_DEFAULT_MODEL_PATH = _DIR / "ggml-silero-v6.2.0.bin"
_CHUNK_SAMPLES: Final = 512
_CHUNK_BYTES: Final = _CHUNK_SAMPLES * 2  # 16-bit

_LOGGER = logging.getLogger()


class InvalidChunkSizeError(Exception):
    """Error raised when chunk size is not correct."""


class SileroVoiceActivityDetector:
    """Detects speech/silence using Silero VAD.

    https://github.com/snakers4/silero-vad
    """

    def __init__(self, model_path: Union[str, Path] = _DEFAULT_MODEL_PATH) -> None:
        self._vctx = _load_model(str(Path(model_path).absolute()))

    @staticmethod
    def chunk_samples() -> int:
        """Return number of samples required for an audio chunk."""
        return _CHUNK_SAMPLES

    @staticmethod
    def chunk_bytes() -> int:
        """Return number of bytes required for an audio chunk."""
        return _CHUNK_BYTES

    def reset(self) -> None:
        """Reset state."""
        _reset(self._vctx)

    def __call__(self, audio: Union[bytes, bytearray, memoryview]) -> float:
        """Return probability of speech [0-1] in a single audio chunk.

        Audio *must* be 512 samples of 16Khz 16-bit mono PCM.
        """
        return self.process_chunk(audio)

    def process_chunk(self, audio: Union[bytes, bytearray, memoryview]) -> float:
        """Return probability of speech [0-1] in a single audio chunk.

        Audio *must* be 512 samples of 16Khz 16-bit mono PCM.
        """
        if len(audio) != _CHUNK_BYTES:
            # Window size is fixed at 512 samples in v5
            raise InvalidChunkSizeError

        audio_array = [sample / _MAX_WAV for sample in array("h", audio)]

        return self.process_samples(audio_array)

    def process_samples(self, samples: Sequence[float]) -> float:
        """Return probability of speech [0-1] in a single audio chunk.

        Audio *must* be 512 float samples [0-1] of 16Khz mono.
        """
        if len(samples) != _CHUNK_SAMPLES:
            # Window size is fixed at 512 samples in v5
            raise InvalidChunkSizeError

        return _process_chunk(self._vctx, samples)

    def process_chunks(
        self, audio: Union[bytes, bytearray, memoryview]
    ) -> Iterable[float]:
        """Return probability of speech in audio [0-1] for each chunk of audio.

        Audio must be 16Khz 16-bit mono PCM.
        """
        if len(audio) < _CHUNK_BYTES:
            # Window size is fixed at 512 samples in v5
            raise InvalidChunkSizeError

        num_audio_bytes = len(audio)
        audio_idx = 0

        while (audio_idx + _CHUNK_BYTES) < num_audio_bytes:
            yield self.process_chunk(audio[audio_idx : audio_idx + _CHUNK_BYTES])
            audio_idx += _CHUNK_BYTES
