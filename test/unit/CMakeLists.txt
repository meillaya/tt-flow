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

# Circular-buffer allocator test (todo 28). Links the scheduler lib (which carries
# the shared L1-budget constants + CBAllocator and reuses the schema lib PUBLIC).
add_executable(CBAllocatorTest CBAllocatorTest.cpp)
target_link_libraries(CBAllocatorTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME cb_allocator COMMAND CBAllocatorTest)

# NoC traffic estimator test (todo 29). Links the scheduler lib (which carries
# the NoC estimator + weight planner and reuses the schema lib PUBLIC).
add_executable(NoCEstimatorTest NoCEstimatorTest.cpp)
target_link_libraries(NoCEstimatorTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME noc_estimator COMMAND NoCEstimatorTest)

# Plan-report assembler test (todo 30). Links the scheduler lib (which carries
# buildPlanReport/formatPlanReport + all four planners and reuses the schema lib
# PUBLIC for plan::toJson).
add_executable(PlanReportTest PlanReportTest.cpp)
target_link_libraries(PlanReportTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME plan_report COMMAND PlanReportTest)

# Rounding-contract test (todo 38, D14). Sim/Rounding.h is header-only, so the
# test only needs the include path (no MLIR/LLVM link). It writes
# ${CMAKE_BINARY_DIR}/rounding_consistency.txt for the Python bit-for-bit
# consistency handshake (python/tensixflow/tests/test_rounding.py).
add_executable(RoundingTest RoundingTest.cpp)
target_include_directories(RoundingTest PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/tensixflow/Sim
)
target_compile_definitions(RoundingTest PRIVATE
  TENSIXFLOW_ROUNDING_OUT_DIR="${CMAKE_BINARY_DIR}"
)
target_link_libraries(RoundingTest PRIVATE
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME rounding COMMAND RoundingTest)

# Functional tile simulator test (todo 39, D14). Links the Sim library (which walks
# the schedule IR and PUBLIC-links MLIRTensixflow) plus the MLIR libs needed to parse
# tiny funcs and run the plan -> allocate -> lower pipeline that builds the schedule
# IR the simulator executes. ctest name EXACTLY tile_simulator.
add_executable(TileSimulatorTest TileSimulatorTest.cpp)
target_link_libraries(TileSimulatorTest PRIVATE
  MLIRTensixflowSim
  MLIRTensixflow
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME tile_simulator COMMAND TileSimulatorTest)

# Static performance-model test (todo 42). Links the Analysis lib (which carries
# estimatePerf/formatPerfReport and PUBLIC-links the scheduler lib — NoC estimator
# + L1-budget constants + plan structs — and the schema lib for plan::toJson /
# parsePlan). ctest name EXACTLY perf_model.
add_executable(PerfModelTest PerfModelTest.cpp)
target_link_libraries(PerfModelTest PRIVATE
  MLIRTensixflowAnalysis
  MLIRTensixflowScheduler
  MLIRTensixflowSchema
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME perf_model COMMAND PerfModelTest)
