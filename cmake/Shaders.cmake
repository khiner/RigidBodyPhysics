find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(Xcrun xcrun REQUIRED)
execute_process(COMMAND ${Xcrun} --find metal RESULT_VARIABLE MetalStatus OUTPUT_QUIET ERROR_QUIET)
if(NOT MetalStatus EQUAL 0)
    message(FATAL_ERROR "Install Apple's Metal Toolchain with: xcodebuild -downloadComponent MetalToolchain")
endif()
option(RBP_SAFE_MATH "Disable fast shader math for rounding diagnostics" OFF)
try_run(GpuArchitectureStatus GpuArchitectureCompiled
    ${CMAKE_CURRENT_BINARY_DIR}/gpu-architecture
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpuArchitecture.cpp
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/lib/metal-cpp"
    LINK_LIBRARIES "-framework Metal" "-framework Foundation"
    RUN_OUTPUT_VARIABLE GpuArchitecture
    COMPILE_OUTPUT_VARIABLE GpuArchitectureErrors)
if(NOT GpuArchitectureCompiled OR NOT GpuArchitectureStatus EQUAL 0)
    message(FATAL_ERROR "Cannot detect the local Metal GPU architecture: ${GpuArchitectureErrors}${GpuArchitecture}")
endif()
string(STRIP "${GpuArchitecture}" GpuArchitecture)
message(STATUS "RBP GPU architecture: ${GpuArchitecture}")
set(ShaderDir ${CMAKE_CURRENT_BINARY_DIR}/gen)
set(ShaderArgs)
if(RBP_SAFE_MATH)
    list(APPEND ShaderArgs --safe-math)
endif()
file(GLOB ShaderInputs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${ShaderInputs} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Shaders.py ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FullStep.py
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpuArchitecture.cpp)
execute_process(COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Shaders.py
    --source ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu --output ${ShaderDir}
    --compiler ${CMAKE_CXX_COMPILER} ${ShaderArgs} COMMAND_ERROR_IS_FATAL ANY)
include(${ShaderDir}/Shaders.cmake)
foreach(Name IN LISTS ShaderNames)
    add_custom_command(OUTPUT ${ShaderDir}/${Name}.air
        COMMAND ${Xcrun} -sdk macosx metal -std=metal4.0 -Wno-unused-function -fmetal-math-mode=${ShaderMath_${Name}}
            -c ${ShaderDir}/${Name}.metal -o ${ShaderDir}/${Name}.air
        DEPENDS ${ShaderDir}/${Name}.metal ${ShaderDir}/Shaders.cmake VERBATIM)
    list(APPEND ShaderAir ${ShaderDir}/${Name}.air)
endforeach()
add_custom_command(OUTPUT ${ShaderDir}/rbp.metallib
    COMMAND ${Xcrun} metallib ${ShaderAir} -o ${ShaderDir}/rbp.metallib
    DEPENDS ${ShaderAir} VERBATIM)
file(GENERATE OUTPUT ${ShaderDir}/ArchiveTarget.txt CONTENT "${GpuArchitecture}\n")
add_custom_command(OUTPUT ${ShaderDir}/rbp.binary.metallib
    COMMAND ${Xcrun} metal-tt -target air64-apple-macos26.0 -arch ${GpuArchitecture}
        -L ${ShaderDir} ${ShaderDir}/rbp.mtlp-json -o ${ShaderDir}/rbp.binary.metallib
    WORKING_DIRECTORY ${ShaderDir}
    DEPENDS ${ShaderDir}/rbp.metallib ${ShaderDir}/rbp.mtlp-json ${ShaderDir}/ArchiveTarget.txt VERBATIM)
add_custom_target(rbp_shaders DEPENDS ${ShaderDir}/rbp.binary.metallib)
set_property(TARGET rbp_shaders PROPERTY RBP_SHADER_DIR ${ShaderDir})

# Copy both compiled assets next to an executable, or into an app's Resources directory.
function(rbp_copy_shaders target)
    get_target_property(dir rbp_shaders RBP_SHADER_DIR)
    get_target_property(bundle ${target} MACOSX_BUNDLE)
    if(bundle)
        set(destination "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources")
    else()
        set(destination "$<TARGET_FILE_DIR:${target}>")
    endif()
    # Ninja resolves generated shader files through the link dependencies.
    if(NOT CMAKE_GENERATOR MATCHES "Ninja")
        add_dependencies(${target} rbp_shaders)
    endif()
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${dir}/rbp.metallib ${dir}/rbp.binary.metallib)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${destination}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dir}/rbp.metallib ${dir}/rbp.binary.metallib ${destination}
        VERBATIM)
endfunction()
