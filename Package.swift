// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MetalPathTracer",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "pathtracer",
            path: "Sources/PathTracer",
            resources: [
                .copy("Resources/pathtrace.metal")
            ],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ]
        )
    ]
)
