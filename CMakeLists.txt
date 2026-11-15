cmake_minimum_required(VERSION 3.26)
project(pysilero_vad VERSION 3.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Python - use SABIModule since this extension uses Py_LIMITED_API (Stable ABI)
find_package(Python COMPONENTS Development.Module Development.SABIModule REQUIRED)

# CPU instructions are picked at runtime where GGML can: its CPU backend is
# built once per instruction level (x86-64: SSE2, SSE4.2, AVX, AVX2, AVX-512...;
# Linux ARM64: ARMv8.0 up to SVE/SME) and the best one the CPU has is loaded
# when the module is imported. A single non-native build would bake in one
# level: GGML_NATIVE=OFF still means AVX2/FMA/F16C on x86-64, which crashed
# with SIGILL on CPUs without them (Celeron, Atom, pre-Haswell).
#
# 32-bit x86 can't build the AVX2 variant (64-bit pdep), and GGML has no
# variants for ARM outside Linux/macOS: those get one baseline build.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(PYSILERO_X86 ON)
endif()
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND (PYSILERO_X86 OR (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")))
    set(PYSILERO_CPU_VARIANTS ON)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    # Where GGML writes the backend modules it builds (it requires one).
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(PYSILERO_X86)
    foreach(feat SSE42 AVX AVX2 FMA F16C BMI2)
        set(GGML_${feat} OFF CACHE BOOL "" FORCE)
    endforeach()
endif()

# Build GGML as a subdirectory
# This will respect all GGML_* options passed via cmake.define in pyproject.toml
add_subdirectory(src/ggml)

# The CPU backend's libraries: one per variant, or the one ggml-cpu.
if(PYSILERO_CPU_VARIANTS)
    get_property(_ggml_targets DIRECTORY src/ggml/src PROPERTY BUILDSYSTEM_TARGETS)
    set(GGML_CPU_TARGETS)
    foreach(_target IN LISTS _ggml_targets)
        get_target_property(_type ${_target} TYPE)
        if(_target MATCHES "^ggml-cpu-" AND _type STREQUAL "MODULE_LIBRARY")
            list(APPEND GGML_CPU_TARGETS ${_target})
        endif()
    endforeach()
    message(STATUS "pysilero-vad CPU backends: ${GGML_CPU_TARGETS}")
else()
    set(GGML_CPU_TARGETS ggml-cpu)
endif()

# Create the Python extension module using the Stable ABI (limited API).
# USE_SABI 3.9 makes CMake target Py_LIMITED_API=0x03090000 and produce an
# `.abi3.so` suffix, so a single wheel is importable on all CPython 3.9+.
Python_add_library(silero_vad MODULE WITH_SOABI USE_SABI 3.9
    src/py_silero_vad.cpp
)

# Link against Python Stable ABI and GGML libraries. With the variants, the
# CPU backend isn't linked: load_backends() loads it from the package.
target_link_libraries(silero_vad PRIVATE
    Python::SABIModule
    ggml
    ggml-base
)
if(NOT PYSILERO_CPU_VARIANTS)
    target_link_libraries(silero_vad PRIVATE ggml-cpu)
endif()

# Include directories
target_include_directories(silero_vad PRIVATE
    src
    src/ggml/include
    src/ggml/src
    src/ggml/src/ggml-cpu
)

# Set the output name and location
set_target_properties(silero_vad PROPERTIES
    OUTPUT_NAME "silero_vad"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pysilero_vad"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Platform-specific RPATH settings and library installation
if(WIN32)
    # Windows doesn't use RPATH - DLLs are searched in the same directory as the extension
    # Install GGML DLLs directly alongside the Python extension
    set(GGML_LIB_DEST pysilero_vad)
elseif(APPLE)
    # macOS uses @loader_path
    set_target_properties(silero_vad PROPERTIES
        INSTALL_RPATH "@loader_path/lib"
        BUILD_RPATH "@loader_path/lib"
    )
    set(GGML_LIB_DEST pysilero_vad/lib)
    # The GGML libs also depend on each other (e.g. ggml-cpu needs
    # ggml-base), so each one needs its own rpath to find its siblings in
    # the same directory. Relying solely on silero_vad's rpath only works
    # at runtime via transitive lookup, which auditwheel's dependency
    # resolver does not perform when repairing the wheel.
    set_target_properties(ggml ggml-base ${GGML_CPU_TARGETS} PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH "@loader_path"
        BUILD_RPATH "@loader_path"
    )
else()
    # Linux uses $ORIGIN
    set_target_properties(silero_vad PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_RPATH "$ORIGIN/lib"
    )
    target_link_options(silero_vad PRIVATE "-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib")
    set(GGML_LIB_DEST pysilero_vad/lib)
    # See the APPLE branch above for why the GGML libs need their own rpath.
    set_target_properties(ggml ggml-base ${GGML_CPU_TARGETS} PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
endif()

# Add version and commit info
target_compile_definitions(silero_vad PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install the module
install(TARGETS silero_vad
    LIBRARY DESTINATION pysilero_vad
    RUNTIME DESTINATION pysilero_vad  # For Windows DLLs
    COMPONENT python_modules
)

# Install GGML shared libraries alongside the Python extension
install(TARGETS ggml ggml-base ${GGML_CPU_TARGETS}
    LIBRARY DESTINATION ${GGML_LIB_DEST}
    RUNTIME DESTINATION ${GGML_LIB_DEST}  # For Windows DLLs
    COMPONENT python_modules
)

# Install Python package files (__init__.py, model file, etc.)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/pysilero_vad/
    DESTINATION pysilero_vad
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.bin"
    PATTERN "__pycache__" EXCLUDE
)
