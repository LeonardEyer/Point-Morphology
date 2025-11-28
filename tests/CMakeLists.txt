enable_testing()

# === Fetch GoogleTest ===
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(project_tests
    test_InverseIterative.cpp
)

target_link_libraries(project_tests PRIVATE gtest_main pointmorph)

include(GoogleTest)
gtest_discover_tests(project_tests)
