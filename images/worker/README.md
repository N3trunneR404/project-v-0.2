# Worker Images

Use Docker Buildx with QEMU emulation to produce multi-architecture images:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --name dt-builder --use

# Native amd64 image
docker buildx build -f Dockerfile.native -t dt/worker-native:latest .

# ARM64 via QEMU
docker buildx build -f Dockerfile.qemu.arm64 -t dt/worker-qemu-arm64:latest --platform linux/amd64 .

# RISC-V via QEMU
docker buildx build -f Dockerfile.qemu.riscv64 -t dt/worker-qemu-riscv64:latest --platform linux/amd64 .

# Optional: publish a multi-arch manifest
docker buildx imagetools create -t dt/worker:latest \
    dt/worker-native:latest \
    dt/worker-qemu-arm64:latest \
    dt/worker-qemu-riscv64:latest
```

The Wasm flavour (`Dockerfile.wasm`) expects a containerd shim such as `wasmtime` or `wasmedge` and is published separately if required.

