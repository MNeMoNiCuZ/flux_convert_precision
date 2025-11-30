# Flux Model Precision Converter

A tool for converting Flux AI models between different precision formats (FP32, FP16, BF16, FP8). This tool simplifies precision conversion with an intuitive interface and robust memory management.

## Features
- **Multiple Precisions**: Support for FP32, FP16, BF16, and FP8 formats
- **Interactive Mode**: User-friendly guided interface for batch conversions
- **CLI Mode**: Command-line interface for scripting and automation
- **Memory Efficient**: Automatic memory-efficient mode for large models (>20GB)
- **Smart Filename Handling**: Preserves precision tag casing (e.g., `_FP16` → `_BF16`)
- **Batch Processing**: Convert multiple files to multiple precisions in one session
- **Results Summary**: Detailed conversion results with file sizes and status

## System Requirements

- **Python**: 3.13.2 (tested) or compatible version
- **OS**: Windows (Linux/macOS compatible with minor adjustments)
- **RAM**: Recommended 32GB+ for large model conversions
- **Storage**: Sufficient space for input and output models

## Setup

1.  **Create Virtual Environment**: Create a virtual environment manually, or run the `venv_create.bat` script to automatically create a Python virtual environment. It will also offer to install the required packages.
2.  **Install Dependencies**: If you didn't install the packages in the previous step, activate the environment (`venv\Scripts\activate.bat`) and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Interactive Mode (Recommended)

Simply run the script without arguments to enter interactive mode:

```bash
python app.py
```

You'll be guided through:
1. **File Selection**: Paste one or more model paths (supports `.safetensors` auto-detection)
2. **Precision Detection**: Automatically detects input precision from model files
3. **Output Selection**: Choose target precisions from a formatted table
4. **Batch Conversion**: Converts all files automatically with progress tracking

### CLI Mode

For automation and scripting:

```bash
python app.py --input_model model.safetensors --output_model model_fp16.safetensors --save_precision fp16
```

**CLI Arguments:**
- `--input_model`: Path to input Flux model (safetensors file)
- `--output_model`: Path to save the converted model (safetensors file)
- `--save_precision`: Target precision (see [Supported Precisions](#supported-precisions))
- `--mem_eff`: Use memory efficient loading/saving (recommended for large models)

**Example:**
```bash
python app.py --input_model flux_dev.safetensors --output_model flux_dev_fp8.safetensors --save_precision fp8 --mem_eff
```

## Supported Precisions

| Format | Alias | File Size (Flux Dev) | Description |
|--------|-------|---------------------|-------------|
| **FP32** | `float`, `float32`, `fp32` | ~44.3 GB | 32-bit floating point (highest quality) |
| **FP16** | `fp16`, `float16` | ~22 GB | 16-bit floating point |
| **BF16** | `bf16`, `bfloat16` | ~22 GB | Brain float 16 (better range than FP16) |
| **FP8** | `fp8`, `float8`, `fp8_e4m3fn` | ~11 GB | 8-bit floating point (smallest size) |

**Additional FP8 variants:**
- `fp8_e5m2`: 8-bit floating point E5M2
- `fp8_e4m3fnuz`: 8-bit floating point E4M3 (FN unsigned zero)
- `fp8_e5m2fnuz`: 8-bit floating point E5M2 (FN unsigned zero)

## Performance

Precision conversion is primarily I/O-bound (reading/writing large files). CPU performance is excellent for this task:

- **FP16 → FP32** (22GB model): ~5-10 seconds
- **Bottleneck**: Disk speed, not CPU compute
- **Memory**: Uses system RAM

## Advanced Features

### Filename Casing Preservation

The tool automatically detects and preserves the casing style of precision tags in filenames:

- `MyModel_FP16.safetensors` → `MyModel_BF16.safetensors` (uppercase preserved)
- `mymodel_fp16.safetensors` → `mymodel_bf16.safetensors` (lowercase preserved)

### Batch Conversion

In interactive mode, you can:
- Convert **multiple files** at once
- Convert to **multiple precisions** in one session
- Track results with a detailed summary table

## File Structure

```
flux_convert_precision/
├── app.py              # User interface (interactive + CLI)
├── converter.py        # Core conversion logic
├── requirements.txt    # Python dependencies
├── venv_create.bat     # Virtual environment setup script
└── README.md           # This file
```

## Troubleshooting

**Issue**: "File not found" error
- **Solution**: Ensure the file path is correct. The tool supports both absolute and relative paths.


## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Credits
- [Original code](https://github.com/bmaltais/kohya_ss)

Built with:
- [PyTorch](https://pytorch.org/)
- [Safetensors](https://github.com/huggingface/safetensors)
- [tqdm](https://github.com/tqdm/tqdm)
- [colorama](https://github.com/tartley/colorama)
