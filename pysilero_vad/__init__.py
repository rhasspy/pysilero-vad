import logging
from pathlib import Path
from typing import Final, Iterable, Union

import numpy as np
import onnxruntime

_RATE: Final = 16000  # Khz
_MAX_WAV: Final = 32767
_DIR = Path(__file__).parent
_DEFAULT_ONNX_PATH = _DIR / "models" / "silero_vad.onnx"
_CONTEXT_SIZE: Final = 64  # 16Khz
_CHUNK_SAMPLES: Final = 512
_CHUNK_BYTES: Final = _CHUNK_SAMPLES * 2  # 16-bit

_LOGGER = logging.getLogger()


class InvalidChunkSizeError(Exception):
    """Error raised when chunk size is not correct."""


class SileroVoiceActivityDetector:
    """Detects speech/silence using Silero VAD.

    https://github.com/snakers4/silero-vad
    """

    def __init__(self, onnx_path: Union[str, Path] = _DEFAULT_ONNX_PATH) -> None:
        onnx_path = str(onnx_path)

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"], sess_options=opts
        )

        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(_RATE, dtype=np.int64)

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
        self._state = np.zeros((2, 1, 128)).astype("float32")

    def __call__(self, audio: bytes) -> float:
        """Return probability of speech [0-1] in a single audio chunk.

        Audio *must* be 512 samples of 16Khz 16-bit mono PCM.
        """
        return self.process_chunk(audio)

    def process_chunk(self, audio: bytes) -> float:
        """Return probability of speech [0-1] in a single audio chunk.

        Audio *must* be 512 samples of 16Khz 16-bit mono PCM.
        """
        if len(audio) != _CHUNK_BYTES:
            # Window size is fixed at 512 samples in v5
            raise InvalidChunkSizeError

        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / _MAX_WAV

        # Add batch dimension and context
        audio_array = np.concatenate(
            (self._context, audio_array[np.newaxis, :]), axis=1
        )
        self._context = audio_array[:, -_CONTEXT_SIZE:]

        # ort_inputs = {"input": audio_array, "state": self._state, "sr": self._sr}
        ort_inputs = {
            "input": audio_array[:, : _CHUNK_SAMPLES + _CONTEXT_SIZE],
            "state": self._state,
            "sr": self._sr,
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._state = ort_outs

        return out.squeeze()

    def process_chunks(self, audio: bytes) -> Iterable[float]:
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
