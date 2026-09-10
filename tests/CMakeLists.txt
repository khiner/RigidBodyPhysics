add_executable(RbpTests
    ContactTest.cpp
    DerivativeTest.cpp
    HullTest.cpp
    LayoutTest.cpp
    Main.cpp
    MeshTest.cpp
    SolveTest.cpp
    WorldTest.cpp
)
target_link_libraries(RbpTests PRIVATE rbp)
target_include_directories(RbpTests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/lib/doctest)
add_test(NAME RbpTests COMMAND RbpTests)
