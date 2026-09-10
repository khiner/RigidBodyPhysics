add_executable(RbpTests
    BroadPhaseTest.cpp
    ContactTest.cpp
    CylinderTest.cpp
    DerivativeTest.cpp
    DistanceTest.cpp
    HullTest.cpp
    LayoutTest.cpp
    Main.cpp
    MeshTest.cpp
    SolveBlockTest.cpp
    SolveTest.cpp
    WorldTest.cpp
)
target_link_libraries(RbpTests PRIVATE rbp)
target_include_directories(RbpTests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/lib/doctest)
add_test(NAME RbpTests COMMAND RbpTests)
