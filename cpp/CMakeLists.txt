# CMake configuration for NL2Trail C++ inference engine
cmake_minimum_required(VERSION 3.16)
project(nl2trail_production CXX)

# Set C++17 standard (required for ONNX Runtime C++ API)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")  # Native CPU optimizations

# ===== Dependency Detection =====

# Option 1: Try vcpkg first (recommended)
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    
    # Find ONNX Runtime via vcpkg
    find_package(Microsoft.ML.OnnxRuntime CONFIG QUIET)
    if(Microsoft.ML.OnnxRuntime_FOUND)
        set(ONNX_LIB Microsoft.ML.OnnxRuntime)
        set(ONNX_FOUND TRUE)
        message(STATUS "Found ONNX Runtime via vcpkg")
    endif()
    
    # Find SentencePiece via vcpkg
    find_package(unofficial-sentencepiece CONFIG QUIET)
    if(unofficial-sentencepiece_FOUND)
        set(SPM_LIB unofficial::sentencepiece::sentencepiece)
        set(SPM_FOUND TRUE)
        message(STATUS "Found SentencePiece via vcpkg")
    endif()
else()
    # Option 2: Manual installation paths
    set(ONNXRUNTIME_DIR "C:/onnxruntime" CACHE STRING "Path to ONNX Runtime")
    set(SENTENCEPIECE_DIR "C:/sentencepiece" CACHE STRING "Path to SentencePiece")
    
    # Find ONNX Runtime manually
    if(EXISTS ${ONNXRUNTIME_DIR})
        message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
        include_directories(${ONNXRUNTIME_DIR}/include)
        link_directories(${ONNXRUNTIME_DIR}/lib)
        set(ONNX_FOUND TRUE)
        set(ONNX_LIB "onnxruntime")
    else()
        set(ONNX_FOUND FALSE)
        message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
    endif()
    
    # Find SentencePiece manually
    if(EXISTS ${SENTENCEPIECE_DIR})
        message(STATUS "Using SentencePiece from: ${SENTENCEPIECE_DIR}")
        include_directories(${SENTENCEPIECE_DIR}/include)
        link_directories(${SENTENCEPIECE_DIR}/lib)
        set(SPM_FOUND TRUE)
        set(SPM_LIB "sentencepiece")
    else()
        set(SPM_FOUND FALSE)
        message(WARNING "SentencePiece not found at ${SENTENCEPIECE_DIR}")
    endif()
endif()

# Check if dependencies are available
if(NOT ONNX_FOUND)
    message(FATAL_ERROR "
Missing ONNX Runtime dependency. Please install:

Option 1 - Manual Installation (Recommended):
  1. Download ONNX Runtime: https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip
  2. Extract to C:/onnxruntime
  3. Configure: cmake ..

Current status:
  ONNX Runtime: ${ONNX_FOUND}
  SentencePiece: ${SPM_FOUND} (will use simple tokenizer if not found)
")
endif()

# Use simple tokenizer if SentencePiece not found
if(NOT SPM_FOUND)
    message(STATUS "SentencePiece not found - using simple tokenizer implementation")
    add_definitions(-DUSE_SIMPLE_TOKENIZER)
    set(SPM_LIB "")
endif()

# ===== Project Configuration =====
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/NL2Trail.cpp
    src/SpmTokenizer.cpp
)

# Create library
add_library(nl2trail_lib ${SOURCES})

# Link libraries
target_link_libraries(nl2trail_lib ${ONNX_LIB} ${SPM_LIB})

# Main executable
add_executable(nl2trail src/main.cpp)
target_link_libraries(nl2trail nl2trail_lib)

# ===== Platform-specific Configuration =====
if(WIN32 AND NOT ONNX_FOUND STREQUAL "vcpkg")
    # Copy ONNX Runtime DLL to output directory
    if(EXISTS ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll)
        add_custom_command(TARGET nl2trail POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:nl2trail>
            COMMENT "Copying ONNX Runtime DLL"
        )
    endif()
endif()

# ===== Build Summary =====
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime: ${ONNX_FOUND}")
message(STATUS "SentencePiece: ${SPM_FOUND}")
message(STATUS "Link Libraries: ${ONNX_LIB} ${SPM_LIB}")
message(STATUS "======================================")
