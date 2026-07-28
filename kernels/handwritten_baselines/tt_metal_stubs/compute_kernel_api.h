// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY umbrella for the tt-metal compute kernel API, pinned to
// tt-metal @df36cd6. COMPILE CONTRACT ONLY — pulls in the matmul / register /
// pack intrinsic stubs. Kernels may include either this umbrella or the
// individual compute_kernel_api/*.h headers.

#pragma once

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
