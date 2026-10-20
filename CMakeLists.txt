cmake_minimum_required(VERSION 3.16)
project(mrag LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release: CPU/GPU inference is highly sensitive to optimization flags.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Optional CUDA build ----
# Usage: cmake -S . -B build -DMRAG_CUDA=ON
option(MRAG_CUDA "Build llama.cpp with CUDA support" OFF)

include(FetchContent)

# ---- llama.cpp build options ----
# Derived from MRAG_CUDA rather than hard-coded, so that -DMRAG_CUDA=ON actually works.
# (A plain `set(GGML_CUDA OFF CACHE ... FORCE)` would silently override the command line.)
set(GGML_CUDA            ${MRAG_CUDA} CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          ON  CACHE BOOL "" FORCE)

if(MRAG_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "MRAG_CUDA=ON but CMAKE_CUDA_ARCHITECTURES is unset; CMake will auto-detect.")
    message(STATUS "For RTX 50-series (Blackwell, sm_120) pass -DCMAKE_CUDA_ARCHITECTURES=120")
endif()

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        b11528
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(llama nlohmann_json)

add_executable(mrag
    src/main.cc
    src/mragapp.cc
    src/config.cc
    src/chunk.cc
    src/document.cc
    src/llamamodel.cc
    src/embeddingengine.cc
    src/generationengine.cc
    src/vectordatabase.cc
)

target_include_directories(mrag PRIVATE src)

target_link_libraries(mrag PRIVATE
    llama
    ggml
    nlohmann_json::nlohmann_json
)
