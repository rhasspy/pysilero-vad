"""Setup for pysilero-vad."""

import platform
import os
from pathlib import Path

# Available at setup time due to pyproject.toml
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

_DIR = Path(__file__).parent
_SRC_DIR = _DIR / "src"
_GGML_DIR = _SRC_DIR / "ggml"
_GGML_SRC_DIR = _GGML_DIR / "src"

version = "3.1.0"

# -----------------------------------------------------------------------------


# Monkey patch the compiler because Mac has to be different 🫠
class BuildExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler

        if hasattr(compiler, "_compile"):
            # GCC/Clang (Linux, MacOS)
            orig_compile = compiler._compile  # save original

            def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                extra = list(extra_postargs or [])
                is_cpp = src.endswith((".cpp", ".cc", ".cxx"))

                # Only add C++20 for C++ sources
                if is_cpp:
                    # CXX
                    if not any(a.startswith("-std=c++") for a in extra):
                        extra.append("-std=c++20")

                # Call the original compiler with our tweaked flags
                return orig_compile(obj, src, ext, cc_args, extra, pp_opts)

            # Monkeypatch the compiler
            compiler._compile = new_compile

        # Now run the normal build_ext logic
        super().build_extensions()


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

if os.name == "nt":
    # Assume MSVC on Windows
    extra_compile_args = ["/O2", "/std:c++20"]
    libraries = ["advapi32"]  # for Reg* crap
else:
    # Assume GCC/Clang on Linux/MacOS
    extra_compile_args = [
        "-O3",
        "-Wno-unused-function",
        "-ffast-math",
        "-fno-math-errno",
        "-fno-finite-math-only",
    ]
    libraries = []

    if arch == "x86":
        # Assume SSE4.2 baseline
        extra_compile_args += [
            "-msse4.2",  # enables SSE2/SSE3/SSSE3/SSE4.1 as well
            "-mpopcnt",  # part of the de-facto x86-64-v2 baseline
            "-mtune=generic",  # good across Intel/AMD
        ]

ext_modules = [
    Extension(
        name="pysilero_vad.silero_vad",
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
        extra_compile_args=extra_compile_args,
        libraries=libraries,
    ),
]

setup(version=version, ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
