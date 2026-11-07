//===- DeviceSpecsTest.cpp --------------------------------------*- C++ -*-===//
//
// Unit tests for the generated device table (include/tensixflow/Generated/
// DeviceSpecs.h). The expected values are the published figures recorded in
// spec/devices.json; these tests are the C++ side of the generator --check
// contract.
//
//   1. LookupResolvesEveryName - every name and alias resolves to its entry;
//                                unknown names return nullptr.
//   2. WormholeValues          - n150/n300 grids, L1, clock, FLOPs, DRAM, NoC.
//   3. BlackholeValues         - p100a/p150a grids, L1, clock, DRAM, power.
//   4. CBudgetArithmetic       - cbBudget is usable L1 minus the reserve, never
//                                negative, and the default depth fits every
//                                device.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/Generated/DeviceSpecs.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace mlir::tensixflow::device;

namespace {

constexpr int64_t kBytesPerGiB = int64_t{1} << 30;
constexpr int64_t kDefaultThreeCBs = 3;

TEST(DeviceSpecsTest, LookupResolvesEveryName) {
  EXPECT_EQ(allDevices().size(), 4u);
  EXPECT_EQ(defaultDeviceName(), "n150");
  EXPECT_EQ(validDeviceNames(), "n150, n300, p100a, p150a");

  for (const DeviceSpec &spec : allDevices()) {
    EXPECT_EQ(lookup(spec.name), &spec) << "device " << spec.name;
    EXPECT_NE(spec.arch, nullptr);
  }

  // The generated default is a real table entry.
  ASSERT_NE(defaultDevice(), nullptr);
  EXPECT_EQ(defaultDevice()->name, "n150");
}

TEST(DeviceSpecsTest, LookupHandlesAliasesAndRejectsUnknown) {
  // Aliases resolve to the same entry as the canonical name.
  EXPECT_EQ(lookup("n150d"), lookup("n150"));
  EXPECT_EQ(lookup("n150s"), lookup("n150"));
  EXPECT_EQ(lookup("n300d"), lookup("n300"));
  EXPECT_EQ(lookup("n300s"), lookup("n300"));
  EXPECT_EQ(lookup("bh_p100a"), lookup("p100a"));
  EXPECT_EQ(lookup("bh_p150a"), lookup("p150a"));

  // Unknown names are rejected rather than defaulted.
  EXPECT_EQ(lookup("nope"), nullptr);
  EXPECT_EQ(lookup(""), nullptr);
  EXPECT_EQ(lookup("n150x"), nullptr);
}

TEST(DeviceSpecsTest, WormholeValues) {
  const DeviceSpec *n150 = lookup("n150");
  ASSERT_NE(n150, nullptr);
  EXPECT_STREQ(n150->name, "n150");
  EXPECT_STREQ(n150->arch, "wormhole_b0");
  EXPECT_EQ(n150->computeGrid.gridX, 8);
  EXPECT_EQ(n150->computeGrid.gridY, 8);
  EXPECT_EQ(n150->computeGrid.chips, 1);
  EXPECT_EQ(n150->computeGrid.totalCores(), 64);
  EXPECT_EQ(n150->l1UsableBytes, 1499136);
  EXPECT_EQ(n150->l1NominalBytes, 1572864);
  EXPECT_EQ(n150->tileHeight, 32);
  EXPECT_EQ(n150->tileWidth, 32);
  EXPECT_EQ(n150->tileBytes, 2048);
  EXPECT_EQ(n150->clockHz, 1000000000);
  EXPECT_EQ(n150->flopsPerCyclePerCore, 4096);
  EXPECT_EQ(n150->dramBytes, 12 * kBytesPerGiB);
  EXPECT_EQ(n150->dramBandwidthBytesPerSecond, 288000000000);
  EXPECT_EQ(n150->nocCount, 2);
  EXPECT_EQ(n150->nocFlitBytes, 32);
  EXPECT_EQ(n150->nocMaxBurstBytes, 8192);
  EXPECT_EQ(n150->nocBytesPerCyclePerDirection, 32);
  EXPECT_EQ(n150->powerWatts, 160);

  const DeviceSpec *n300 = lookup("n300");
  ASSERT_NE(n300, nullptr);
  EXPECT_STREQ(n300->arch, "wormhole_b0");
  EXPECT_EQ(n300->computeGrid.gridX, 8);
  EXPECT_EQ(n300->computeGrid.gridY, 7);
  EXPECT_EQ(n300->computeGrid.chips, 2);
  EXPECT_EQ(n300->computeGrid.totalCores(), 112);
  EXPECT_EQ(n300->l1UsableBytes, n150->l1UsableBytes);
  EXPECT_EQ(n300->clockHz, n150->clockHz);
  EXPECT_EQ(n300->dramBytes, 24 * kBytesPerGiB);
  EXPECT_EQ(n300->dramBandwidthBytesPerSecond, 576000000000);
  EXPECT_EQ(n300->powerWatts, 300);
}

TEST(DeviceSpecsTest, BlackholeValues) {
  const DeviceSpec *p100a = lookup("p100a");
  ASSERT_NE(p100a, nullptr);
  EXPECT_STREQ(p100a->arch, "blackhole");
  EXPECT_EQ(p100a->computeGrid.gridX, 12);
  EXPECT_EQ(p100a->computeGrid.gridY, 10);
  EXPECT_EQ(p100a->computeGrid.totalCores(), 120);
  EXPECT_EQ(p100a->l1UsableBytes, 1572864);
  EXPECT_EQ(p100a->l1NominalBytes, 1572864);
  EXPECT_EQ(p100a->clockHz, 1350000000);
  EXPECT_EQ(p100a->flopsPerCyclePerCore, 4096);
  EXPECT_EQ(p100a->dramBytes, 28 * kBytesPerGiB);
  EXPECT_EQ(p100a->dramBandwidthBytesPerSecond, 448000000000);
  EXPECT_EQ(p100a->powerWatts, 300);

  const DeviceSpec *p150a = lookup("p150a");
  ASSERT_NE(p150a, nullptr);
  EXPECT_STREQ(p150a->arch, "blackhole");
  EXPECT_EQ(p150a->computeGrid.totalCores(), 120);
  EXPECT_EQ(p150a->dramBytes, 32 * kBytesPerGiB);
  EXPECT_EQ(p150a->dramBandwidthBytesPerSecond, 512000000000);

  // The boards differ only in DRAM capacity and bandwidth in this model.
  EXPECT_EQ(p100a->l1UsableBytes, p150a->l1UsableBytes);
  EXPECT_EQ(p100a->clockHz, p150a->clockHz);
}

TEST(DeviceSpecsTest, CBudgetArithmetic) {
  for (const DeviceSpec &spec : allDevices()) {
    const int64_t budget = spec.cbBudget(kDefaultL1ReserveBytes);
    EXPECT_EQ(budget, spec.l1UsableBytes - kDefaultL1ReserveBytes)
        << "device " << spec.name;
    EXPECT_GT(budget, 0) << "device " << spec.name;

    // The default depth for the default three-CB pipeline must fit every
    // device; otherwise the product could not run the planned pipeline at all.
    const int64_t defaultBytes =
        kDefaultThreeCBs * kDefaultCBDepthTiles * spec.tileBytes;
    EXPECT_LE(defaultBytes, budget) << "device " << spec.name;
    EXPECT_GE(budget / (kDefaultThreeCBs * spec.tileBytes),
              kDefaultCBDepthTiles)
        << "device " << spec.name;
  }

  // A reserve larger than usable L1 clamps at zero instead of going negative.
  const DeviceSpec *n150 = lookup("n150");
  ASSERT_NE(n150, nullptr);
  EXPECT_EQ(n150->cbBudget(n150->l1UsableBytes), 0);
  EXPECT_EQ(n150->cbBudget(n150->l1UsableBytes + 1), 0);

  // The per-core budget is the published usable L1, not the fabricated 1 MiB.
  EXPECT_EQ(n150->cbBudget(0), 1499136);
}

} // namespace
