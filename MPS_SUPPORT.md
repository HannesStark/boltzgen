# MPS (Metal Performance Shaders) Support for BoltzGen

This document describes the MPS support that has been added to BoltzGen, enabling the use of PyTorch on Apple Silicon (M1, M2, M3, etc.) GPUs.

## What Changed

The repository has been updated to support PyTorch's MPS (Metal Performance Shaders) backend, which allows BoltzGen to run on Apple Silicon GPUs alongside the existing CUDA and CPU support.

### Key Changes

1. **Device Utility Module** (`src/boltzgen/utils/device.py`)
   - Added centralized device detection and management
   - Automatically detects the best available device: CUDA > MPS > CPU
   - Provides device-agnostic cache clearing and autocast support

2. **CLI Updates** (`src/boltzgen/cli/boltzgen.py`)
   - Replaced hardcoded CUDA device detection with device-agnostic functions
   - `get_device_capability()` - works for CUDA/MPS/CPU
   - `get_device_count()` - returns correct device count for all backends

3. **Model Updates** (`src/boltzgen/model/models/boltz.py`)
   - Updated all `torch.autocast("cuda", ...)` calls to use `get_autocast_device_type()`
   - Replaced `torch.cuda.empty_cache()` with `empty_cache()`
   - Updated device tensor creation to use `get_device_type()`

4. **Validation Updates** (`src/boltzgen/model/validation/refolding.py`)
   - Updated cache clearing to support MPS
   - Added conditional CUDA-specific cleanup (only runs on CUDA devices)

5. **Module Updates** (`src/boltzgen/model/modules/trunk.py`)
   - Updated autocast calls in template and token distance modules

## Usage

### Running on Apple Silicon (MPS)

BoltzGen will automatically detect and use MPS when running on Apple Silicon:

```bash
# No special flags needed - MPS will be auto-detected
boltzgen run design_spec.yaml --output results/
```

### Device Selection Priority

The device selection follows this priority:
1. **CUDA** - If NVIDIA GPU is available
2. **MPS** - If Apple Silicon GPU is available
3. **CPU** - Fallback

### Checking Device

You can verify which device is being used by checking the logs during execution. The CLI will print:
```
Using kernels: True/False [device capability: (X, Y)]
Using N devices
```

### Configuration Files

The YAML configuration files use `accelerator: gpu` which works for both CUDA and MPS:
- PyTorch Lightning automatically detects the appropriate GPU backend
- No changes needed to existing config files

## Limitations and Considerations

### MPS vs CUDA Performance

1. **Single Device**: MPS currently supports only a single device, while CUDA can use multiple GPUs
2. **Kernel Support**: Some CUDA-specific kernels may not be available on MPS
3. **Memory Management**: MPS memory management differs from CUDA; you may need to adjust batch sizes

### Known Issues

1. **Mixed Precision**: MPS autocast support requires PyTorch 2.1+
2. **Some Operations**: A few operations may fall back to CPU on MPS
3. **Memory**: MPS shares memory with the system, unlike dedicated CUDA GPUs

## Requirements

- **PyTorch**: 2.0+ (2.1+ recommended for full MPS autocast support)
- **macOS**: 12.3+ (Monterey or later)
- **Apple Silicon**: M1, M2, M3, or later

## Testing

To verify MPS support is working:

```python
import torch
from boltzgen.utils.device import get_device_type, get_device_count

print(f"Device type: {get_device_type()}")  # Should print "mps" on Apple Silicon
print(f"Device count: {get_device_count()}")  # Should print 1 on MPS
print(f"MPS available: {torch.backends.mps.is_available()}")  # Should be True
```

## Migration Notes

If you have custom code or scripts that reference CUDA explicitly:

### Before
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
```

### After
```python
from boltzgen.utils.device import get_device_type, empty_cache

device = get_device_type()
empty_cache()
```

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes on MPS and adjust based on available memory
2. **Precision**: Use `bf16-mixed` precision (already configured) for best performance
3. **Kernels**: The `--use_kernels` flag works automatically based on device capability

## Troubleshooting

### "MPS backend out of memory"
- Reduce batch size in config files
- Close other applications to free up memory
- MPS shares system memory, unlike dedicated GPUs

### Slower than expected
- Ensure PyTorch 2.1+ is installed for optimal MPS support
- Check that `torch.backends.mps.is_available()` returns `True`
- Some operations may still fall back to CPU

### Import errors
- Verify PyTorch installation: `pip install torch>=2.1.0`
- Check macOS version: `sw_vers` (should be 12.3+)

## Contributing

When adding new PyTorch code:
1. Use `boltzgen.utils.device` functions instead of hardcoded device strings
2. Use `get_autocast_device_type()` for autocast contexts
3. Use `empty_cache()` instead of `torch.cuda.empty_cache()`
4. Test on both CUDA and MPS if possible

## References

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch Lightning MPS Support](https://lightning.ai/docs/pytorch/stable/accelerators/mps.html)
