#!/bin/bash
set -euo pipefail

# Build worker images and DT API for heterogeneous compute fabric
# Usage: ./scripts/build-images.sh [--import-to-k3d CLUSTER_NAME] [--api-only|--workers-only]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER_DIR="$PROJECT_ROOT/images/worker"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

IMAGES_TO_IMPORT=()
BUILD_API=true
BUILD_WORKERS=true

# Parse optional flags
for arg in "$@"; do
    case "$arg" in
        --api-only)
            BUILD_WORKERS=false
            ;;
        --workers-only)
            BUILD_API=false
            ;;
    esac
done

# Build DT API image
if [[ "$BUILD_API" == "true" ]]; then
    echo "Building DT API image..."
    cd "$PROJECT_ROOT"
    echo "  Building dt/dt-api:latest..."
    docker build -f deploy/Dockerfile.dt-api -t dt/dt-api:latest .
    IMAGES_TO_IMPORT+=("dt/dt-api:latest")
    echo "✔ DT API image built successfully."
    echo ""
fi

# Build worker images
if [[ "$BUILD_WORKERS" == "true" ]]; then
    cd "$WORKER_DIR"
    echo "Building worker images..."

    # Build native x86_64 image
    echo "  Building dt/worker-native:latest..."
    docker build -f Dockerfile.native -t dt/worker-native:latest .
    IMAGES_TO_IMPORT+=("dt/worker-native:latest")

    # Build QEMU ARM64 image (built for amd64 platform but contains QEMU emulator)
    echo "  Building dt/worker-qemu-arm64:latest..."
    docker build -f Dockerfile.qemu.arm64 -t dt/worker-qemu-arm64:latest .
    IMAGES_TO_IMPORT+=("dt/worker-qemu-arm64:latest")

    # Build QEMU RISC-V image (built for amd64 platform but contains QEMU emulator)
    echo "  Building dt/worker-qemu-riscv64:latest..."
    docker build -f Dockerfile.qemu.riscv64 -t dt/worker-qemu-riscv64:latest .
    IMAGES_TO_IMPORT+=("dt/worker-qemu-riscv64:latest")

    echo "✔ All worker images built successfully."
fi

echo ""
echo "Images:"
docker images | grep "^dt/" || true

# Import to k3d if requested
if [[ "${1:-}" == "--import-to-k3d" ]] && [[ -n "${2:-}" ]]; then
    CLUSTER_NAME="$2"
    echo ""
    echo "Importing images to k3d cluster '$CLUSTER_NAME'..."
    k3d image import "${IMAGES_TO_IMPORT[@]}" -c "$CLUSTER_NAME"
    echo "✔ Images imported to k3d cluster."
fi

