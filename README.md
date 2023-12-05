# pySilero VAD

A pre-packaged voice activity detector using [silero-vad](https://github.com/snakers4/silero-vad).

``` sh
pip install pysilero-vad
```

``` python
from pysilero_vad import SileroVoiceActivityDetector

vad = SileroVoiceActivityDetector()

# Audio must be 16Khz, 16-bit mono PCM
if vad(audio_bytes) >= 0.5:
    print("Speech")
else:
    print("Silence")
```

