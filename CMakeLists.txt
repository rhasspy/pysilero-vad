cmake_minimum_required(VERSION 3.14)
project(pysilero_vad VERSION 3.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Python - use SABIModule since this extension uses Py_LIMITED_API (Stable ABI)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.SABIModule)

# Build GGML as a subdirectory
# This will respect all GGML_* options passed via cmake.define in pyproject.toml
add_subdirectory(src/ggml)

# Create the Python extension module using Stable ABI
Python3_add_library(silero_vad MODULE WITH_SOABI
    src/py_silero_vad.cpp
)

# Link against GGML libraries
target_link_libraries(silero_vad PRIVATE ggml ggml-base ggml-cpu)

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
else()
    # Linux uses $ORIGIN
    set_target_properties(silero_vad PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_RPATH "$ORIGIN/lib"
    )
    target_link_options(silero_vad PRIVATE "-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib")
    set(GGML_LIB_DEST pysilero_vad/lib)
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
install(TARGETS ggml ggml-base ggml-cpu
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
