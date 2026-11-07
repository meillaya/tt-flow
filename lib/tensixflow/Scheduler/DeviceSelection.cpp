//===- DeviceSelection.cpp --------------------------------------*- C++ -*-===//
//
// Driver-level device selection. The command line option lives here so every
// driver that links this library sees the same value, and the passes read the
// selected spec from one global rather than from command line state.
//
// The option parser rejects an unknown device name at parse time and prints the
// list of valid names, so a typo can never reach the planners.
//
//===----------------------------------------------------------------------===//

#include "DeviceSelection.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace tensixflow {
namespace device {

namespace {

/// Parser that validates the device name against the generated table.
class DeviceNameParser final : public llvm::cl::parser<std::string> {
public:
  DeviceNameParser(llvm::cl::Option &option)
      : llvm::cl::parser<std::string>(option) {}

  bool parse(llvm::cl::Option &option, llvm::StringRef argName,
             llvm::StringRef arg, std::string &value) {
    (void)argName;
    if (lookup(arg) == nullptr) {
      return option.error("unknown device '" + arg + "'; valid devices: " +
                          validDeviceNames());
    }
    value = arg.str();
    return false;
  }
};

llvm::cl::opt<std::string, /*ExternalStorage=*/false, DeviceNameParser>
    deviceOption(
        "tensixflow-device",
        llvm::cl::desc(
            "Tenstorrent device spec used to plan and to stamp "
            "tensixflow.device on the module (default: n150)"),
        llvm::cl::init(kDefaultDeviceName));

} // namespace

llvm::StringRef selectedDeviceName() { return deviceOption; }

const DeviceSpec *selectedDevice() { return lookup(selectedDeviceName()); }

} // namespace device
} // namespace tensixflow
} // namespace mlir
