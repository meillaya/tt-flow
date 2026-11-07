# Emitted-project check against the pinned TT-Metalium release.
#
# This is the real compile gate. It emits a fresh project from
# examples/matmul.mlir with tensixflow-opt, configures it with CMake against the
# extracted TT-Metalium package in .deps/, and compiles and links the host
# program. scripts/build_generated.sh owns all of that; it prints the fetch
# command and exits 0 with an explicit skip message when .deps/ is absent, and
# that skip is never reported as a pass.
#
# Run it explicitly:
#   devenv shell -- bash scripts/fetch_tt_metal.sh
#   cmake --build build --target check-generated-project

set(TENSIXFLOW_GENERATED_CHECK_SCRIPT
    "${PROJECT_SOURCE_DIR}/scripts/build_generated.sh")

add_custom_target(check-generated-project
  COMMAND ${CMAKE_COMMAND} -E env
          "TENSIXFLOW_BUILD_DIR=${PROJECT_BINARY_DIR}"
          bash "${TENSIXFLOW_GENERATED_CHECK_SCRIPT}"
  DEPENDS tensixflow-opt
  COMMENT "Emitting a project and building it against the pinned TT-Metalium release"
  VERBATIM
)
set_target_properties(check-generated-project PROPERTIES FOLDER "Tests")
