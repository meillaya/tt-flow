//===- DeviceSelection.h ----------------------------------------*- C++ -*-===//
//
// Driver-level device selection for the scheduling passes.
//
// Every driver that links MLIRTensixflowScheduler registers the
// --tensixflow-device=<name> command line option from here. The passes read the
// selected device from this one place instead of re-parsing command line state.
// The option defaults to the generated default device (n150), and the parser
// rejects an unknown name with the list of valid names.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_DEVICESELECTION_H
#define TENSIXFLOW_SCHEDULER_DEVICESELECTION_H

#include "tensixflow/Generated/DeviceSpecs.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tensixflow {
namespace device {

/// Discardable module attribute that records the selected device name on the
/// module. The planning passes stamp it so later passes and the JSON emitter
/// agree on one device without re-reading command line state.
inline constexpr const char kDeviceAttrName[] = "tensixflow.device";

/// Device name selected on the command line, or the generated default when the
/// option is absent. Never empty.
llvm::StringRef selectedDeviceName();

/// Spec for selectedDeviceName(). Never null: the option parser rejects unknown
/// names, and the default name is generated from spec/devices.json.
const DeviceSpec *selectedDevice();

} // namespace device
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_DEVICESELECTION_H
