import CoreML
import Foundation

@available(macOS 14.4, *)
private struct OperationRecord {
    let name: String
    let cost: Double?
    let preferred: MLComputeDevice?
}

@available(macOS 14.4, *)
private func collect(
    block: MLModelStructure.Program.Block,
    plan: MLComputePlan,
    into records: inout [OperationRecord]
) {
    for operation in block.operations {
        records.append(
            OperationRecord(
                name: operation.operatorName,
                cost: plan.estimatedCost(of: operation)?.weight,
                preferred: plan.deviceUsage(for: operation)?.preferred
            )
        )
        for nested in operation.blocks {
            collect(block: nested, plan: plan, into: &records)
        }
    }
}

private func fail(_ message: String, code: Int32 = 2) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(code)
}

@main
private struct PlacementInspector {
    static func main() async {
        guard #available(macOS 14.4, *) else {
            fail("MLComputePlan requires macOS 14.4 or newer")
        }
        guard CommandLine.arguments.count == 3 else {
            fail("usage: inspect_placement MODEL.mlmodelc MIN_ANE_RATE")
        }
        let path = CommandLine.arguments[1]
        guard path.hasSuffix(".mlmodelc") else {
            fail("model path must end in .mlmodelc: \(path)")
        }
        guard let minimum = Double(CommandLine.arguments[2]),
              minimum >= 0.0, minimum <= 1.0 else {
            fail("MIN_ANE_RATE must be in [0,1]")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            fail("compiled model directory does not exist: \(path)")
        }

        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let plan = try await MLComputePlan.load(
                contentsOf: URL(fileURLWithPath: path, isDirectory: true),
                configuration: configuration
            )
            guard case .program(let program) = plan.modelStructure else {
                fail("expected an ML Program model structure")
            }
            guard let main = program.functions["main"] else {
                fail("ML Program has no main function")
            }
            var records: [OperationRecord] = []
            collect(block: main.block, plan: plan, into: &records)

            var costTotal = 0.0
            var costAne = 0.0
            var costCpu = 0.0
            var costGpu = 0.0
            var costUnknown = 0.0
            var preferredAne = 0
            var preferredCpu = 0
            var preferredGpu = 0
            var preferredUnknown = 0
            var cpuOps: [String] = []
            var unknownOps: [String] = []

            for record in records {
                switch record.preferred {
                case .neuralEngine?: preferredAne += 1
                case .cpu?:
                    preferredCpu += 1
                    cpuOps.append(record.name)
                case .gpu?: preferredGpu += 1
                case .some(_):
                    preferredUnknown += 1
                    unknownOps.append(record.name)
                case nil:
                    preferredUnknown += 1
                    unknownOps.append(record.name)
                }
                guard let cost = record.cost else { continue }
                costTotal += cost
                switch record.preferred {
                case .neuralEngine?: costAne += cost
                case .cpu?: costCpu += cost
                case .gpu?: costGpu += cost
                case .some(_): costUnknown += cost
                case nil: costUnknown += cost
                }
            }

            guard costTotal > 0.0 else {
                fail("MLComputePlan returned no estimated operation cost", code: 1)
            }
            let aneRate = costAne / costTotal
            // Cost weights are documented as fractions of total model
            // execution. Unknown preferred devices with non-zero workload make
            // the placement verdict incomplete and therefore fail closed.
            let complete = costUnknown <= 1.0e-9
            let pass = complete && aneRate >= minimum
            let opAssessed = preferredAne + preferredCpu + preferredGpu
            let opRate = opAssessed == 0 ? 0.0 : Double(preferredAne) / Double(opAssessed)

            print("format=vokra-coreml-placement-v1")
            print("compute_units=cpu_and_neural_engine")
            print("operations_total=\(records.count)")
            print("operations_ane_preferred=\(preferredAne)")
            print("operations_cpu_preferred=\(preferredCpu)")
            print("operations_gpu_preferred=\(preferredGpu)")
            print("operations_unknown=\(preferredUnknown)")
            print(String(format: "operation_count_ane_rate=%.9f", opRate))
            print(String(format: "estimated_cost_total=%.9f", costTotal))
            print(String(format: "estimated_cost_ane=%.9f", costAne))
            print(String(format: "estimated_cost_cpu=%.9f", costCpu))
            print(String(format: "estimated_cost_gpu=%.9f", costGpu))
            print(String(format: "estimated_cost_unknown=%.9f", costUnknown))
            print(String(format: "ane_placement_rate=%.9f", aneRate))
            print(String(format: "minimum_ane_placement_rate=%.9f", minimum))
            print("cpu_preferred_operators=\(cpuOps.sorted().joined(separator: ","))")
            print("unknown_preferred_operators=\(unknownOps.sorted().joined(separator: ","))")
            print("verdict=\(pass ? "PASS" : "FAIL")")
            if !pass { exit(1) }
        } catch {
            fail("failed to construct CoreML compute plan: \(error)", code: 1)
        }
    }
}
