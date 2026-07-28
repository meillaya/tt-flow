// SPDX-License-Identifier: Apache-2.0
//
// NEGATIVE compile fixture — deliberately STALE.
//
// This file references `EnqueueProgram`, a LEGACY tt-metal symbol that was
// REMOVED in favor of the current distributed mesh API
// (distributed::EnqueueMeshWorkload). Because the stub headers (pinned to
// tt-metal @df36cd6) do NOT declare EnqueueProgram, this file MUST FAIL to
// compile with `-fsyntax-only` ("'EnqueueProgram' was not declared").
//
// That failure is the point: it proves the compile check is NON-VACUOUS. If
// the stubs ever re-declared a removed legacy symbol (or otherwise drifted from
// the current API), this file would start compiling and the
// check-generated-kernels-compile target would fail its negative assertion.
//
// DO NOT "fix" this file. It is supposed to be broken.

#include "host_api.hpp"

namespace host {

void run_legacy() {
    using namespace tt::tt_metal;
    Program program = CreateProgram();
    // Legacy symbol — intentionally undeclared in the current-API stubs.
    EnqueueProgram(program);
}

}  // namespace host
