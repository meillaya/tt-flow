// SPDX-License-Identifier: Apache-2.0
//
// Hand-written single-core matmul BASELINE, current tt-metal API (@df36cd6).
//
// This is the POSITIVE half of the non-vacuous compile check: it genuinely
// references the CURRENT host + kernel symbols and MUST compile
// `-fsyntax-only` against the stub headers in tt_metal_stubs/. (It is never
// linked or run — there is no device and no tt-metal library; see README
// "Status / honesty".)
//
// The emitter (todos 34-36) will produce kernels shaped like this; if a future
// edit to the stubs drops a current symbol this file references, the
// check-generated-kernels-compile target fails loudly.

#include <cstdint>
#include <memory>
#include <vector>

#include "compute_kernel_api.h"
#include "dataflow_api.h"
#include "host_api.hpp"

// ---------------------------------------------------------------------------
// KERNEL SIDE (runs on a Tensix core; compiled with the kernel intrinsics).
// ---------------------------------------------------------------------------
namespace compute {

// Single-tile C += A * B over one K-tile, using the current intrinsic set.
void kernel_main() {
    // Compile-time args (set by the host via ComputeConfig.compile_args).
    constexpr auto in0_cb = static_cast<std::uint32_t>(tt::CBIndex::c_0);
    constexpr auto in1_cb = static_cast<std::uint32_t>(tt::CBIndex::c_1);
    constexpr auto out_cb = static_cast<std::uint32_t>(tt::CBIndex::c_16);
    const auto num_tiles = get_compile_time_arg_val<std::uint32_t>(0);

    matmul_init(/*in0_block_c_tiles=*/1, /*in1_block_r_tiles=*/1);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(in0_cb, 1);
        cb_wait_front(in1_cb, 1);

        tile_regs_acquire();
        matmul_tiles(in0_cb, in1_cb, /*in0_tile=*/0, /*in1_tile=*/0,
                     /*dst_tile=*/0, /*transpose_of_faces=*/false);
        tile_regs_commit();

        cb_reserve_back(out_cb, 1);
        tile_regs_wait();
        pack_tile(/*tile_index=*/0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);

        cb_pop_front(in0_cb, 1);
        cb_pop_front(in1_cb, 1);
    }
}

}  // namespace compute

// ---------------------------------------------------------------------------
// DATAMOVEMENT SIDE (reader/writer NoC intrinsics).
// ---------------------------------------------------------------------------
namespace dataflow {

void reader_main() {
    const auto dst_x = get_arg_val<std::uint32_t>(0);
    const auto dst_y = get_arg_val<std::uint32_t>(1);
    const auto src_l1 = get_arg_val<std::uint32_t>(2);
    const auto dst_l1 = get_arg_val<std::uint32_t>(3);
    const auto bytes = get_arg_val<std::uint32_t>(4);

    const std::uint64_t src_noc = get_noc_addr(dst_x, dst_y, src_l1);
    (void)src_noc;
    noc_async_read(src_l1, dst_l1, bytes);
    noc_async_read_barrier();

    const std::uint64_t mcast =
        get_noc_multicast_addr(0, 0, 7, 7, dst_l1);
    (void)mcast;
    noc_async_write(dst_l1, dst_l1, bytes);
    noc_async_write_barrier();

    noc_semaphore_inc(get_noc_addr(dst_x, dst_y, dst_l1), 1);
    (void)get_write_ptr(static_cast<std::uint32_t>(tt::CBIndex::c_0));
    (void)get_read_ptr(static_cast<std::uint32_t>(tt::CBIndex::c_0));
}

}  // namespace dataflow

// ---------------------------------------------------------------------------
// HOST SIDE (builds + enqueues the program; the current mesh Enqueue* API).
// ---------------------------------------------------------------------------
namespace host {

void run() {
    using namespace tt::tt_metal;

    auto device = distributed::MeshDevice::create_unit_mesh(/*device_id=*/0);

    Program program = CreateProgram();

    // Compute kernel: note the SINGLE current CreateKernel + config variant.
    ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .compile_args = {32},
    };
    KernelHandle compute_kernel =
        CreateKernel(program, "kernels/compute.cpp", CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}},
                     compute_config);

    // Reader (data-movement) kernel via the same variant entry point.
    DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = {},
    };
    KernelHandle reader_kernel =
        CreateKernel(program, "kernels/reader.cpp", CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}},
                     reader_config);

    // Circular buffers addressed by tt::CBIndex.
    CreateCircularBuffer(
        program, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}},
        CircularBufferConfig(2 * 32 * 32 * 2,
                             {{tt::CBIndex::c_0, 32 * 32 * 2},
                              {tt::CBIndex::c_1, 32 * 32 * 2},
                              {tt::CBIndex::c_16, 32 * 32 * 2}}));

    // DRAM input buffer (interleaved).
    auto in0 = CreateBuffer(device.get(),
                            InterleavedBufferConfig{
                                .size = 32 * 32 * 2,
                                .page_size = 32 * 32 * 2,
                                .buffer_type = BufferType::DRAM,
                            });

    SetCommonRuntimeArgs(program, compute_kernel, {32});
    SetRuntimeArgs(program, reader_kernel, CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}},
                   {0, 0, 0, 0, 32 * 32 * 2});

    // Enqueue via the current distributed mesh API.
    distributed::CommandQueue cq{};
    distributed::MeshWorkload workload{};
    std::vector<std::uint32_t> src(32 * 32, 0);
    distributed::EnqueueWriteMeshBuffer(cq, in0, src);
    distributed::EnqueueMeshWorkload(cq, device, workload, /*blocking=*/true);
    std::vector<std::uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, in0);
}

}  // namespace host
