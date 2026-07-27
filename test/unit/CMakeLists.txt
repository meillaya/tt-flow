# C++ unit-test suite (GoogleTest). Deliberately SEPARATE from the lit/FileCheck
# IR suite in test/ (which defines check-tensixflow). Tests register with ctest
# via add_test, so `ctest -R <name>` runs them.

find_package(GTest REQUIRED)
message(STATUS "Using GTest: ${GTest_DIR}")

add_executable(PlanSchemaTest PlanSchemaTest.cpp)
target_link_libraries(PlanSchemaTest PRIVATE
  MLIRTensixflowSchema
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)
# Absolute path to the source schema/ dir so the test validates against
# schema/plan.schema.json regardless of the ctest working directory.
target_compile_definitions(PlanSchemaTest PRIVATE
  TENSIXFLOW_SCHEMA_DIR="${PROJECT_SOURCE_DIR}/schema"
)

add_test(NAME plan_schema COMMAND PlanSchemaTest)
