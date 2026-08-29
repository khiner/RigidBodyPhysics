add_executable(RbpTests Main.cpp HullTest.cpp MeshTest.cpp LayoutTest.cpp MathTest.cpp SolveTest.cpp WorldTest.cpp)
target_link_libraries(RbpTests PRIVATE rbp)
target_include_directories(RbpTests SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/doctest)
add_test(NAME RbpTests COMMAND RbpTests)
