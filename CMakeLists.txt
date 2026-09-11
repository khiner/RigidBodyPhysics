cmake_minimum_required(VERSION 3.20)

# Embedding hosts choose their own toolchain.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RBP_STANDALONE ON)
else()
    set(RBP_STANDALONE OFF)
endif()

if(RBP_STANDALONE)
    # Select Homebrew LLVM before project() detects the compiler.
    execute_process(COMMAND brew --prefix llvm OUTPUT_VARIABLE HomebrewLlvm OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_COMPILER "${HomebrewLlvm}/bin/clang++" CACHE FILEPATH "")
endif()

project(RigidBodyPhysics LANGUAGES CXX)

if(NOT APPLE)
    message(FATAL_ERROR "RigidBodyPhysics targets macOS on Apple Silicon.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(RBP_STANDALONE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd
    set(CMAKE_CXX_FLAGS "-stdlib=libc++")
    # Allow partial designated initialization.
    add_compile_options(-Wall -Wextra -Wno-missing-field-initializers)
    add_link_options("-L${HomebrewLlvm}/lib/c++" "-Wl,-rpath,${HomebrewLlvm}/lib/c++")
endif()

# Compile metal-cpp definitions in exactly one translation unit per program.
option(RBP_METAL_CPP_IMPL "Compile the metal-cpp private implementation into rbp" ON)
option(RBP_TOOLS "Build the engine's scene and bench tools and its tests" ${RBP_STANDALONE})

include(cmake/Shaders.cmake)

add_library(rbp STATIC
    src/metal/Context.cpp
    src/Hull.cpp
    src/Mesh.cpp
    src/Solver.cpp
    src/World.cpp
)
if(RBP_METAL_CPP_IMPL)
    target_sources(rbp PRIVATE src/metal/MetalCppImpl.cpp)
endif()
# Preserve the embedding workaround for the compiler issue reproduced by probes/VectorMiscompile.cpp.
target_compile_options(rbp PRIVATE -O1)
target_compile_features(rbp PUBLIC cxx_std_23)
target_include_directories(rbp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_include_directories(rbp SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/metal-cpp)
target_link_libraries(rbp PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

if(RBP_TOOLS)
    foreach(Tool IN ITEMS Scenes Bench Replay)
        add_executable(Rbp${Tool} tools/${Tool}.cpp)
        target_link_libraries(Rbp${Tool} PRIVATE rbp)
        rbp_copy_shaders(Rbp${Tool})
        target_compile_options(Rbp${Tool} PRIVATE -O1)
    endforeach()
    target_include_directories(RbpScenes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_include_directories(RbpBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    enable_testing()
    add_subdirectory(tests)
endif()
