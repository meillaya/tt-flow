// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY stub of the tt-metal HOST API surface, pinned to tt-metal
// @df36cd6 (host_api.hpp L148/164/211/364/405/481/546).
//
// COMPILE CONTRACT ONLY — every body is empty. There is NO behavior and there
// must never be any. Generated host.cpp (todos 34-36) is validated by compiling
// it `-fsyntax-only` against this header.
//
// CURRENT-API SYMBOLS DECLARED HERE (a correct kernel references these):
//   CreateProgram, CreateKernel(Program&, file, core_spec,
//       variant<DataMovementConfig, ComputeConfig>), CreateCircularBuffer,
//   CreateBuffer(InterleavedBufferConfig | ShardedBufferConfig),
//   SetRuntimeArgs, SetCommonRuntimeArgs,
//   distributed::EnqueueWriteMeshBuffer / EnqueueMeshWorkload /
//       EnqueueReadMeshBuffer, distributed::MeshDevice (create_unit_mesh),
//   tt::CBIndex (kernel_types.hpp).
//
// LEGACY SYMBOLS DELIBERATELY OMITTED (a stale kernel referencing these FAILS to
// compile — this is what proves the check is non-vacuous):
//   EnqueueProgram, and the old per-role CreateKernel overloads
//   (CreateDataMovementKernel / CreateComputeKernel). They do NOT exist in
//   current tt-metal; do NOT add them here.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kernel_types.hpp"

namespace tt::tt_metal {

// Forward declaration so CreateBuffer can take a mesh-device pointer without a
// circular include (distributed::MeshDevice is defined further below).
namespace distributed {
class MeshDevice;
}  // namespace distributed

// host_api.hpp L148 — create an (empty) program.
Program CreateProgram();

// host_api.hpp L164 — the SINGLE current CreateKernel: a config variant selects
// data-movement vs compute. (The legacy per-role overloads are intentionally
// absent.)
KernelHandle CreateKernel(Program& program,
                          const std::string& file_name,
                          const CoreSpec& core_spec,
                          const std::variant<DataMovementConfig, ComputeConfig>& config);

// host_api.hpp L211 — allocate a circular buffer on the given cores.
CBHandle CreateCircularBuffer(Program& program,
                              const CoreSpec& core_spec,
                              const CircularBufferConfig& config);

// host_api.hpp L364 / L405 — allocate a DRAM/L1 buffer (interleaved | sharded).
std::shared_ptr<Buffer> CreateBuffer(distributed::MeshDevice* device,
                                     const InterleavedBufferConfig& config);
std::shared_ptr<Buffer> CreateBuffer(distributed::MeshDevice* device,
                                     const ShardedBufferConfig& config);

// host_api.hpp L481 / L546 — per-dispatch and common runtime args.
void SetRuntimeArgs(Program& program,
                    KernelHandle kernel,
                    const CoreSpec& cores,
                    const std::vector<std::uint32_t>& runtime_args);
void SetCommonRuntimeArgs(Program& program,
                          KernelHandle kernel,
                          const std::vector<std::uint32_t>& runtime_args);

// --- Distributed / mesh surface (current Enqueue* API) ----------------------
namespace distributed {

class MeshDevice {
public:
    // Open a 1x1 ("unit") mesh device — the no-hardware modeling entry point.
    static std::shared_ptr<MeshDevice> create_unit_mesh(std::uint32_t device_id = 0);
};

class CommandQueue {};
class MeshWorkload {};

// Enqueue a host->device buffer write.
void EnqueueWriteMeshBuffer(CommandQueue& cq,
                            const std::shared_ptr<Buffer>& buffer,
                            const std::vector<std::uint32_t>& src);

// Enqueue a program (workload) onto the mesh.
void EnqueueMeshWorkload(CommandQueue& cq,
                         const std::shared_ptr<MeshDevice>& device,
                         MeshWorkload& workload,
                         bool blocking = false);

// Enqueue a device->host buffer read.
void EnqueueReadMeshBuffer(CommandQueue& cq,
                           std::vector<std::uint32_t>& dst,
                           const std::shared_ptr<Buffer>& buffer);

}  // namespace distributed
}  // namespace tt::tt_metal
