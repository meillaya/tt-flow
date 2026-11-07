# C++ unit-test suite (GoogleTest). Deliberately separate from the lit/FileCheck
# IR suite in test/ (which defines check-tensixflow). Tests register with ctest
# via add_test, so `ctest -R <name>` runs them.

find_package(GTest REQUIRED)
message(STATUS "Using GTest: ${GTest_DIR}")

# Generated device table test. Header-only, so the test links only LLVMSupport
# (StringRef and ArrayRef) plus GoogleTest.
add_executable(DeviceSpecsTest DeviceSpecsTest.cpp)
target_link_libraries(DeviceSpecsTest PRIVATE
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME device_specs COMMAND DeviceSpecsTest)

# Keep the generated header and Python module in step with spec/devices.json.
# The same check is available as the check-device-spec build target.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME device_spec_generator
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gen_device_spec.py
          --check
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(PlanSchemaTest PlanSchemaTest.cpp)
target_link_libraries(PlanSchemaTest PRIVATE
  MLIRTensixflowSchema
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)
# Absolute path to the source schema/ directory so the test validates against
# schema/plan.schema.json regardless of the ctest working directory.
target_compile_definitions(PlanSchemaTest PRIVATE
  TENSIXFLOW_SCHEMA_DIR="${PROJECT_SOURCE_DIR}/schema"
)

add_test(NAME plan_schema COMMAND PlanSchemaTest)

# Scheduler data-model test. Links the scheduler model + the reused schema lib;
# no MLIR dialect dependency.
add_executable(SchedulerModelTest SchedulerModelTest.cpp)
target_link_libraries(SchedulerModelTest PRIVATE
  MLIRTensixflowScheduler
  MLIRTensixflowSchema
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME scheduler_model COMMAND SchedulerModelTest)

# Core-grid planner test. Links the scheduler lib (grid planner + generated
# device table; reuses the schema lib PUBLIC).
add_executable(GridPlannerTest GridPlannerTest.cpp)
target_link_libraries(GridPlannerTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME grid_planner COMMAND GridPlannerTest)

# Weight-strategy planner test. Links the scheduler lib (weight planner +
# generated device table; reuses the schema lib PUBLIC).
add_executable(WeightPlannerTest WeightPlannerTest.cpp)
target_link_libraries(WeightPlannerTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME weight_planner COMMAND WeightPlannerTest)

# Circular-buffer allocator test. Links the scheduler lib (CB allocator +
# grid planner + generated device table; reuses the schema lib PUBLIC).
add_executable(CBAllocatorTest CBAllocatorTest.cpp)
target_link_libraries(CBAllocatorTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME cb_allocator COMMAND CBAllocatorTest)

# NoC traffic estimator test. Links the scheduler lib (NoC estimator + weight
# planner; reuses the schema lib PUBLIC).
add_executable(NoCEstimatorTest NoCEstimatorTest.cpp)
target_link_libraries(NoCEstimatorTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME noc_estimator COMMAND NoCEstimatorTest)

# Plan-report assembler test. Links the scheduler lib (buildPlanReport /
# formatPlanReport + all four planners) and the schema lib for the JSON seam.
add_executable(PlanReportTest PlanReportTest.cpp)
target_link_libraries(PlanReportTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME plan_report COMMAND PlanReportTest)

# Rounding-contract test. Sim/Rounding.h is header-only, so the test only needs
# the include path (no MLIR/LLVM link). It writes
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

# Functional tile simulator test. Links the Sim library (which walks the
# schedule IR and PUBLIC-links MLIRTensixflow) plus the MLIR libs needed to
# parse tiny funcs and run the plan -> allocate -> lower pipeline that builds
# the schedule IR the simulator executes.
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

# Static performance-model test. Links the Analysis lib (estimatePerf and
# formatPerfReport; PUBLIC-links the scheduler lib and the generated device
# table) and the schema lib for plan::toJson and parsePlan.
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

# Emitted-path accounting test. EmissionEstimator.h is pure data and math over
# the scheduler model, so the test links the scheduler library and LLVMSupport.
add_executable(EmissionEstimatorTest EmissionEstimatorTest.cpp)
target_link_libraries(EmissionEstimatorTest PRIVATE
  MLIRTensixflowScheduler
  LLVMSupport
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME emission_estimator COMMAND EmissionEstimatorTest)
