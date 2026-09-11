add_executable(RbpTests
    AdvanceTest.cpp
    BroadPhaseTest.cpp
    ContactTest.cpp
    CylinderTest.cpp
    DerivativeTest.cpp
    DistanceTest.cpp
    HullTest.cpp
    LayoutTest.cpp
    Main.cpp
    MeshTest.cpp
    QueryStorageTest.cpp
    SolveBlockTest.cpp
    SolveTest.cpp
    WorldTest.cpp
)
target_link_libraries(RbpTests PRIVATE rbp)
target_include_directories(RbpTests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/lib/doctest)
add_test(NAME RbpTests COMMAND RbpTests)

rbp_copy_shaders(RbpTests)
set(EmbeddedText "#pragma once\nnamespace gpu {\n")
foreach(Source IN ITEMS Shared.h Solve.metal BroadPhase.metal Layout.metal)
    get_filename_component(Name ${Source} NAME_WE)
    file(READ "${PROJECT_SOURCE_DIR}/src/gpu/${Source}" Text)
    string(APPEND EmbeddedText "constexpr auto ${Name}Source = R\"MSL(\n${Text})MSL\";\n")
endforeach()
string(APPEND EmbeddedText "}\n")
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GpuSource.h CONTENT "${EmbeddedText}")
target_include_directories(RbpTests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(RBP_SAFE_MATH)
    target_compile_definitions(RbpTests PRIVATE RBP_SAFE_MATH)
endif()
