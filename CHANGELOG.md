# Changelog

## 3.3.0

- Use cmake with full optimizations

## 3.2.0

- Only require SSE2 on x86_64

## 3.1.0

- Require SSE4.1 on x86_64
- Use fast math optimizations on non-Windows platforms

## 3.0.1

- Set C++ std version for non-Linux OSes

## 3.0.0

- Use ggml model instead of onnxruntime and numpy
- Replace `process_array` function with `process_samples`

## 2.1.1

- Add subdirectories to pyproject.toml
- Update dev dependencies
- Clean up extra files

## 2.1.0

- Loosen dependencies
- Add `process_array` function to allow processing numpy floating point audio directly

## 2.0.1

- Migrate to pyproject.toml

## 2.0.0

- Update to Silero VAD v5
- Audio size **must** be 512 samples in `process_chunk`
- Add `process_chunks` to handle longer audio

## 1.0.0

- Initial version
