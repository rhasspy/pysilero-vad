"""Setup for pysilero-vad."""

import platform
from pathlib import Path

# Available at setup time due to pyproject.toml
from setuptools import setup, Extension

_DIR = Path(__file__).parent
_SRC_DIR = _DIR / "src"
_GGML_DIR = _SRC_DIR / "ggml"
_GGML_SRC_DIR = _GGML_DIR / "src"

version = "3.0.1"


# -----------------------------------------------------------------------------

sources = [_SRC_DIR / "py_silero_vad.cpp"]
sources.extend(
    [
        _GGML_SRC_DIR / f
        for f in (
            "ggml-alloc.c",
            "ggml-backend.cpp",
            "ggml-backend-reg.cpp",
            "ggml.c",
            "ggml-opt.cpp",
            "ggml-quants.c",
            "ggml-threading.cpp",
        )
    ]
)

# CPU backend
sources.extend(
    [
        _GGML_SRC_DIR / "ggml-cpu" / f
        for f in (
            "binary-ops.cpp",
            "ggml-cpu-cpp.cpp",  # renamed to avoid .o conflict
            "ggml-cpu.c",
            "hbm.cpp",
            "ops.cpp",
            "quants.c",
            "repack.cpp",
            "traits.cpp",
            "unary-ops.cpp",
            "vec.cpp",
        )
    ]
)
sources.extend([_GGML_SRC_DIR / "ggml-cpu" / "amx" / f for f in ("amx.cpp", "mmq.cpp")])

# Only x86 and arm are supported
machine = platform.machine().lower()
if ("arm" in machine) or ("aarch" in machine):
    arch = "arm"
else:
    arch = "x86"

sources.extend(
    [
        _GGML_SRC_DIR / "ggml-cpu" / "arch" / arch / f
        for f in (
            "cpu-feats.cpp",
            "quants.c",
            "repack.cpp",
        )
    ]
)

# -----------------------------------------------------------------------------

ext_modules = [
    Extension(
        name="pysilero_vad.silero_vad",
        language="c++",
        py_limited_api=True,
        sources=sorted(str(p.relative_to(_DIR)) for p in sources),
        define_macros=[
            ("Py_LIMITED_API", "0x03090000"),
            ("VERSION_INFO", f'"{version}"'),
            ("GGML_VERSION", '"0.9.4"'),
            ("GGML_COMMIT", '"19ceec8eac980403b714d603e5ca31653cd42a3f"'),
            ("GGML_USE_CPU", "1"),
            ("_GNU_SOURCE", "1"),  # CPU_ZERO
            ("POSIX_C_SOURCE", "200809L"),  # threading?
        ],
        include_dirs=[
            str(_GGML_DIR / "include"),
            str(_GGML_SRC_DIR),
            str(_GGML_SRC_DIR / "ggml-cpu"),
            str(_SRC_DIR),
        ],
        extra_compile_args={"c": ["-O3", "-std=c11"], "cxx": ["-O3", "-std=c++17"]},
    ),
]

setup(ext_modules=ext_modules)
