cmake_minimum_required(VERSION 3.20)

# Homebrew LLVM, not Apple clang, and it has to be chosen before project() probes the compiler.
execute_process(COMMAND brew --prefix llvm OUTPUT_VARIABLE HomebrewLlvm OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_COMPILER "${HomebrewLlvm}/bin/clang++" CACHE FILEPATH "")

project(RigidBodyPhysics LANGUAGES CXX)

if(NOT APPLE)
    message(FATAL_ERROR "RigidBodyPhysics targets macOS on Apple Silicon.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Emit build/compile_commands.json for clangd.

# -O1, not -O2: Homebrew clang 22.1.8 miscompiles vector growth at -O2, and the failure is silent.
# probes/VectorMiscompile.cpp is the reproducer, re-checkable with OPT=-O2 ./run VectorMiscompile.
set(CMAKE_CXX_FLAGS "-O1 -stdlib=libc++")
add_link_options("-L${HomebrewLlvm}/lib/c++" "-Wl,-rpath,${HomebrewLlvm}/lib/c++")

# Metal is compiled at runtime from embedded source, which keeps the build off Xcode's Metal
# toolchain. The shared struct header is embedded the same way, so host and device compile one
# definition rather than two that have to be kept in agreement by hand.
set(GpuSources
    ${CMAKE_SOURCE_DIR}/src/gpu/Shared.h
    ${CMAKE_SOURCE_DIR}/src/gpu/Layout.metal
    ${CMAKE_SOURCE_DIR}/src/gpu/Solve.metal
)
set(GeneratedGpuSource ${CMAKE_BINARY_DIR}/gen/GpuSource.h)
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
    src/metal/MetalCppImpl.cpp
    src/Hull.cpp
    src/Mesh.cpp
    src/Solver.cpp
    src/World.cpp
)
target_sources(rbp PRIVATE ${GeneratedGpuSource})
target_include_directories(rbp PUBLIC src ${CMAKE_BINARY_DIR}/gen SYSTEM PUBLIC lib/metal-cpp)
target_link_libraries(rbp PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")
target_compile_options(rbp PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

add_executable(RbpScenes tools/Scenes.cpp)
target_link_libraries(RbpScenes PRIVATE rbp)
target_compile_options(RbpScenes PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

add_executable(RbpBench tools/Bench.cpp)
target_link_libraries(RbpBench PRIVATE rbp)
target_compile_options(RbpBench PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

enable_testing()
add_subdirectory(tests)
