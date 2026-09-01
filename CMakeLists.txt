cmake_minimum_required(VERSION 3.20)

# A host that add_subdirectory()s this one has already chosen its compiler and flags, so every toolchain-wide setting below is standalone-only.
# Per-target settings go on the target.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RBP_STANDALONE ON)
else()
    set(RBP_STANDALONE OFF)
endif()

if(RBP_STANDALONE)
    # Homebrew LLVM, not Apple clang, chosen before project() probes the compiler.
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
    # Designated initializers leave unnamed fields at their defaults on purpose, hence -Wno-missing-field-initializers.
    add_compile_options(-Wall -Wextra -Wno-missing-field-initializers)
    add_link_options("-L${HomebrewLlvm}/lib/c++" "-Wl,-rpath,${HomebrewLlvm}/lib/c++")
endif()

# A program may hold exactly one metal-cpp private-implementation TU, so a host that already has one sets this OFF.
option(RBP_METAL_CPP_IMPL "Compile the metal-cpp private implementation into rbp" ON)
option(RBP_TOOLS "Build the engine's scene and bench tools and its tests" ${RBP_STANDALONE})

# Metal source is embedded here and compiled at runtime, which keeps the build off Xcode's Metal toolchain.
# Shared.h is embedded the same way, so host and device compile one definition instead of two.
set(GpuSources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Shared.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Layout.metal
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
# These sources are read at configure time, so editing a .metal file or Shared.h requires re-running cmake configure.
# A build without that re-run keeps the previously embedded kernel source.
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
# -O1, not -O2: Homebrew clang 22.1.8 silently miscompiles vector growth at -O2.
# probes/VectorMiscompile.cpp reproduces it, re-checkable with OPT=-O2 ./run VectorMiscompile.
# Set on the target because the later flag wins, so an embedding host's -O2 cannot override it.
target_compile_options(rbp PRIVATE -O1)
target_include_directories(rbp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_include_directories(rbp SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/metal-cpp)
target_link_libraries(rbp PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

if(RBP_TOOLS)
    # The tools include tests/ for Shapes.h, shared between a scene and its test.
    add_executable(RbpScenes tools/Scenes.cpp)
    target_link_libraries(RbpScenes PRIVATE rbp)
    target_compile_options(RbpScenes PRIVATE -O1)
    target_include_directories(RbpScenes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    add_executable(RbpBench tools/Bench.cpp)
    target_link_libraries(RbpBench PRIVATE rbp)
    target_compile_options(RbpBench PRIVATE -O1)
    target_include_directories(RbpBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    enable_testing()
    add_subdirectory(tests)
endif()
