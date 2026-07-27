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

# Scheduler data-model test (todo 25). Links the scheduler model + the reused
# schema lib; no MLIR dialect dependency.
add_executable(SchedulerModelTest SchedulerModelTest.cpp)
target_link_libraries(SchedulerModelTest PRIVATE
  MLIRTensixflowScheduler
  MLIRTensixflowSchema
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME scheduler_model COMMAND SchedulerModelTest)

# Core-grid planner test (todo 26). Links the scheduler lib (which carries the
# shared L1-budget constants + GridPlanner and reuses the schema lib PUBLIC).
add_executable(GridPlannerTest GridPlannerTest.cpp)
target_link_libraries(GridPlannerTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME grid_planner COMMAND GridPlannerTest)

# Weight-strategy planner test (todo 27). Links the scheduler lib (which carries
# the shared L1-budget constants + WeightPlanner and reuses the schema lib
# PUBLIC).
add_executable(WeightPlannerTest WeightPlannerTest.cpp)
target_link_libraries(WeightPlannerTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME weight_planner COMMAND WeightPlannerTest)
