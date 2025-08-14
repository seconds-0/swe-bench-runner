# ARM64/Apple Silicon Support

## Overview

SWE-bench Runner now includes full support for ARM64 architecture (Apple Silicon M1/M2/M3). The tool automatically detects your architecture and configures Docker image building accordingly.

## How It Works

### Automatic Detection
When you run SWE-bench Runner on ARM64:
1. The system automatically detects your architecture
2. Sets the Docker namespace to empty (triggering local builds)
3. Shows a clear message about what's happening
4. Builds images locally using the SWE-bench harness

### First Run Experience
```
━━━ ARM64 Architecture Detected (Apple Silicon) ━━━
ℹ️  SWE-bench will build Docker images locally for ARM64
   • First build: 30-60+ minutes per repository
   • Subsequent runs use cached images (fast)
   • Requires ~120GB free disk space
```

### Progress Tracking
During the long initial builds, you'll see real-time progress:
- Build phase detection (Base, Environment, Repository, Dependencies, Instance)
- Spinner with elapsed time
- Clear status updates

## Requirements

- Docker Desktop for Mac with ARM64 support
- ~120GB free disk space for Docker images
- Patience for the first build (30-60+ minutes)

## Performance

| Operation | First Run | Subsequent Runs |
|-----------|-----------|-----------------|
| Image Build | 30-60+ minutes | Cached (instant) |
| Evaluation | Normal speed | Normal speed |
| Memory Usage | ~8-16GB during build | Normal |

## Technical Details

### Implementation
- Detection: `platform.machine()` returns "arm64" on Apple Silicon
- Namespace: Set to empty string to trigger local builds
- Harness: Uses `--namespace none` flag for local building
- Progress: Real-time output parsing with subprocess.Popen

### Docker Images
The tool builds:
1. Base environment image
2. Repository-specific image
3. Instance-specific image with dependencies

Each subsequent instance in the same repository reuses the base layers.

## Troubleshooting

### Build Fails
- Ensure Docker Desktop is running
- Check available disk space (need ~120GB)
- Try with a smaller dataset first (`--dataset lite`)

### Build Takes Too Long
- First build is expected to take 30-60+ minutes
- Use `--count 1` to test with a single instance first
- Consider running overnight for full datasets

### Memory Issues
- Close other applications during build
- Increase Docker Desktop memory allocation
- Use `--max-workers 1` to limit parallelism

## FAQ

**Q: Why does the first run take so long?**
A: SWE-bench doesn't provide pre-built ARM64 images, so we build them locally. This includes compiling dependencies and setting up complete Python environments.

**Q: Are the locally built images compatible with x86_64 images?**
A: They produce the same evaluation results but are architecture-specific. Don't share images between architectures.

**Q: Can I pre-build images for my team?**
A: Yes! After building, you can export and share Docker images with other ARM64 users.

**Q: Will ARM64 support affect evaluation results?**
A: No. The evaluation logic is identical; only the underlying architecture differs.
