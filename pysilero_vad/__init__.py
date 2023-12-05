import logging
from pathlib import Path
from typing import Final, Union

import numpy as np
import onnxruntime

_RATE: Final = 16000
_MAX_WAV: Final = 32767
_DIR = Path(__file__).parent
_DEFAULT_ONNX_PATH = _DIR / "models" / "silero_vad.onnx"

_LOGGER = logging.getLogger()


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

        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def reset(self) -> None:
        """Reset state."""
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def __call__(self, audio: bytes) -> float:
        """Return probability of speech in audio [0-1].

        Audio must be 16Khz 16-bit mono PCM.
        """
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / _MAX_WAV

        # Add batch dimension
        audio_array = np.expand_dims(audio_array, 0)

        ort_inputs = {
            "input": audio_array,
            "h": self._h,
            "c": self._c,
            "sr": np.array(_RATE, dtype=np.int64),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        return out.squeeze()
