# Build the shared Ji scene/runtime once for its standalone runners and tests.
set(JiSource ${CMAKE_CURRENT_SOURCE_DIR})
set(JiTests ${PROJECT_SOURCE_DIR}/tests/ji)
set(JiTools ${PROJECT_SOURCE_DIR}/tools/ji)

add_library(JiSceneRuntime STATIC
    ${JiSource}/Audit.cpp
    ${JiSource}/Metal.cpp
    ${JiSource}/Oracle.cpp
    ${JiSource}/Scenarios.cpp
    ${JiSource}/Scene.cpp)
target_link_libraries(JiSceneRuntime PUBLIC rbp)
target_include_directories(JiSceneRuntime PUBLIC ${JiSource})
target_compile_options(JiSceneRuntime PRIVATE -O1)

add_executable(JiTests ${JiTests}/SolverTest.cpp ${PROJECT_SOURCE_DIR}/tests/Main.cpp)
target_link_libraries(JiTests PRIVATE JiSceneRuntime)
target_include_directories(JiTests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/lib/doctest)
rbp_copy_shaders(JiTests)
add_test(NAME JiTests COMMAND JiTests)

add_executable(JiBenchmark ${JiTools}/Benchmark.cpp)
target_link_libraries(JiBenchmark PRIVATE JiSceneRuntime)
target_compile_options(JiBenchmark PRIVATE -O1)
rbp_copy_shaders(JiBenchmark)
