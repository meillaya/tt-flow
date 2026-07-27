// RUN: tensixflow-opt %s | FileCheck %s
// Verify the tensixflow dialect is registered and a module round-trips.
// CHECK: module
module {}
