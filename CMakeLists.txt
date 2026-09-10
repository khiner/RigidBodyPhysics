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

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GeneratedFullStep ${CMAKE_CURRENT_BINARY_DIR}/gen/FullStep.metal)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FullStep.py
        --source ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu --compiler ${CMAKE_CXX_COMPILER}
        --output ${GeneratedFullStep}
    COMMAND_ERROR_IS_FATAL ANY
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FullStep.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/FullStep.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/FullStepData.h
)
set(GpuSources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Shared.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/SolveCommandData.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/SolveCommands.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Layout.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/BroadPhase.metal
    ${GeneratedFullStep}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Solve.metal
)
set(GeneratedGpuSource ${CMAKE_CURRENT_BINARY_DIR}/gen/GpuSource.h)
set(EmbeddedText "// Generated from src/gpu by CMake. Do not edit.\n#pragma once\n\nnamespace gpu {\n")
foreach(Path IN LISTS GpuSources)
    get_filename_component(Name "${Path}" NAME_WE)
    file(READ "${Path}" Text)
    string(APPEND EmbeddedText "constexpr auto ${Name}Source = R\"MSL(\n${Text})MSL\";\n\n")
endforeach()
string(APPEND EmbeddedText "} // namespace gpu\n")
file(GENERATE OUTPUT ${GeneratedGpuSource} CONTENT "${EmbeddedText}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GpuSources})

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
target_sources(rbp PRIVATE ${GeneratedGpuSource})
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
        target_compile_options(Rbp${Tool} PRIVATE -O1)
    endforeach()
    target_include_directories(RbpScenes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_include_directories(RbpBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    enable_testing()
    add_subdirectory(tests)
endif()
