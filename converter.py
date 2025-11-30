#!/usr/bin/env python3
"""
Flux Model Precision Converter - Core Module

This module contains the core conversion logic for converting Flux models
between different precisions (float32, fp16, bf16, fp8).

Functions:
    - str_to_dtype: Convert string to torch.dtype
    - MemoryEfficientSafeOpen: Memory efficient safetensors reader
    - mem_eff_save_file: Memory efficient safetensors writer
    - normalize_path: Path normalization utility
    - detect_precision_from_filename: Detect precision from filename
    - generate_output_filename: Generate output filename with precision
    - load_model_metadata: Load metadata from safetensors file
    - convert_precision: Main conversion function
"""

import gc
import json
import logging
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions (copied from library/utils.py)
# ============================================================================

def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


class MemoryEfficientSafeOpen:
    """
    Memory efficient safetensors file reader
    (copied from library/utils.py)
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    Memory efficient save file
    (copied from library/utils.py)
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                logger.warning(f"Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    logger.info(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


# ============================================================================
# Path and Filename Utilities
# ============================================================================

def normalize_path(path_str: str) -> str:
    r"""
    Normalize path to use forward slashes and resolve to absolute path.
    Handles forward slashes (/), backslashes (\), and double backslashes (\\).
    
    Args:
        path_str: Path string with any slash type
    
    Returns:
        Normalized path with forward slashes
    """
    # First use pathlib to normalize the path
    normalized = Path(path_str).resolve()
    # Convert to string with forward slashes
    return str(normalized).replace("\\", "/")


def detect_precision_from_filename(filename: str) -> Optional[str]:
    """
    Detect precision format from filename.
    
    Args:
        filename: Filename to analyze
    
    Returns:
        Detected precision string (e.g., "fp16", "bf16", "fp8") or None if not detected
    """
    # Common precision patterns in filenames
    patterns = [
        r'[_\-\.]fp32[_\-\.]',
        r'[_\-\.]float32[_\-\.]',
        r'[_\-\.]fp16[_\-\.]',
        r'[_\-\.]float16[_\-\.]',
        r'[_\-\.]bf16[_\-\.]',
        r'[_\-\.]bfloat16[_\-\.]',
        r'[_\-\.]fp8[_\-\.]',
        r'[_\-\.]float8[_\-\.]',
        # Also check at the end of the filename (before extension)
        r'[_\-]fp32$',
        r'[_\-]float32$',
        r'[_\-]fp16$',
        r'[_\-]float16$',
        r'[_\-]bf16$',
        r'[_\-]bfloat16$',
        r'[_\-]fp8$',
        r'[_\-]float8$',
    ]
    
    # Remove extension for pattern matching
    name_without_ext = os.path.splitext(filename)[0].lower()
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext)
        if match:
            # Extract the precision part
            matched_text = match.group(0).strip('_-.').lower()
            # Normalize to standard format
            if matched_text in ['fp32', 'float32']:
                return 'fp32'
            elif matched_text in ['fp16', 'float16']:
                return 'fp16'
            elif matched_text in ['bf16', 'bfloat16']:
                return 'bf16'
            elif matched_text in ['fp8', 'float8']:
                return 'fp8'
    
    return None


def generate_output_filename(input_path: str, target_precision: str, use_uppercase: bool = None) -> str:
    """
    Generate output filename based on input path and target precision.
    Preserves original filename casing style.
    
    Args:
        input_path: Path to input file
        target_precision: Target precision (e.g., "fp16", "bf16", "fp8")
        use_uppercase: If True, use uppercase; if False, use lowercase; if None, auto-detect
    
    Returns:
        Full path to output file
    """
    input_path = normalize_path(input_path)
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name_without_ext, ext = os.path.splitext(filename)
    
    # Detect current precision in filename (case-insensitive)
    detected_precision = detect_precision_from_filename(filename)
    
    if detected_precision:
        # Determine target casing
        if use_uppercase is None:
            # Auto-detect from input filename
            # Use lookarounds to match precision not surrounded by alphanumeric chars
            # This allows matching _FP16 (where \b would fail because _ is a word char)
            pattern = re.compile(rf'(?<![a-zA-Z0-9]){re.escape(detected_precision)}(?![a-zA-Z0-9])', re.IGNORECASE)
            match = pattern.search(name_without_ext)
            
            if match:
                actual_precision_in_file = match.group(0)
                use_uppercase = actual_precision_in_file.isupper()
            else:
                use_uppercase = False
        
        # Apply casing
        if use_uppercase:
            target_precision_styled = target_precision.upper()
        else:
            target_precision_styled = target_precision.lower()
        
        # Replace detected precision with target precision
        new_name = name_without_ext
        replaced = False
        
        # Try pattern: _PRECISION_ or .PRECISION. or -PRECISION-
        pattern1 = re.compile(rf'([_\-\.])({re.escape(detected_precision)})([_\-\.])', re.IGNORECASE)
        match1 = pattern1.search(new_name)
        if match1:
            new_name = new_name[:match1.start()] + match1.group(1) + target_precision_styled + match1.group(3) + new_name[match1.end():]
            replaced = True
        
        # Try pattern: _PRECISION or -PRECISION at end
        if not replaced:
            pattern2 = re.compile(rf'([_\-])({re.escape(detected_precision)})$', re.IGNORECASE)
            match2 = pattern2.search(new_name)
            if match2:
                new_name = new_name[:match2.start()] + match2.group(1) + target_precision_styled
                replaced = True
        
        # Fallback: just replace the precision string wherever it appears
        if not replaced:
            pattern3 = re.compile(rf'\b{re.escape(detected_precision)}\b', re.IGNORECASE)
            match3 = pattern3.search(new_name)
            if match3:
                new_name = new_name[:match3.start()] + target_precision_styled + new_name[match3.end():]
    else:
        # No precision detected, append target precision (use lowercase as default)
        new_name = f"{name_without_ext}_{target_precision.lower()}"
    
    # Combine directory, new name, and extension
    output_path = os.path.join(directory, new_name + ext)
    return normalize_path(output_path)



# ============================================================================
# Core Conversion Functions
# ============================================================================

def load_model_metadata(file_name):
    """Load metadata from safetensors file"""
    try:
        with safe_open(file_name, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            return metadata if metadata else {}
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
        return {}


def convert_precision(
    input_model: str,
    output_model: str,
    save_precision: str,
    loading_device: str = "cpu",
    working_device: str = "cpu",
    mem_eff_load_save: bool = False,
):
    """
    Convert Flux model precision

    Args:
        input_model: Path to input model file
        output_model: Path to output model file
        save_precision: Target precision (float, fp16, bf16, fp8)
        loading_device: Device to load model on (cpu or cuda)
        working_device: Device to work on (cpu or cuda)
        mem_eff_load_save: Use memory efficient load/save for large models
    """
    logger.info(f"Converting model: {input_model}")
    logger.info(f"Output model: {output_model}")
    logger.info(f"Target precision: {save_precision}")
    logger.info(f"Loading device: {loading_device}")
    logger.info(f"Working device: {working_device}")

    # Parse precision
    save_dtype = str_to_dtype(save_precision)
    if save_dtype is None:
        raise ValueError(f"Invalid save precision: {save_precision}")

    # Load metadata
    logger.info("Loading metadata...")
    metadata = load_model_metadata(input_model)

    # Load model
    logger.info("Loading model...")
    state_dict = {}
    
    if mem_eff_load_save:
        logger.info("Using memory efficient loading...")
        with MemoryEfficientSafeOpen(input_model) as f:
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensor = f.get_tensor(key)
                state_dict[key] = tensor.to(loading_device)
    else:
        state_dict = load_file(input_model, device=loading_device)

    # Detect and log input precision
    if state_dict:
        sample_tensor = next(iter(state_dict.values()))
        if isinstance(sample_tensor, torch.Tensor):
            logger.info(f"Detected input precision: {sample_tensor.dtype}")

    # Convert precision
    logger.info(f"Converting to {save_precision}...")
    for key in tqdm(list(state_dict.keys()), desc="Converting precision"):
        if isinstance(state_dict[key], torch.Tensor) and state_dict[key].dtype.is_floating_point:
            # Move to working device, convert, then back to loading device
            if working_device != loading_device:
                state_dict[key] = state_dict[key].to(working_device)
            state_dict[key] = state_dict[key].to(save_dtype)
            if working_device != loading_device:
                state_dict[key] = state_dict[key].to(loading_device)

    # Save model
    logger.info(f"Saving model to {output_model}...")
    if mem_eff_load_save:
        logger.info("Using memory efficient save...")
        mem_eff_save_file(state_dict, output_model, metadata)
    else:
        save_file(state_dict, output_model, metadata)

    logger.info("Conversion complete!")
    
    # CRITICAL: Clean up memory to prevent accumulation across multiple conversions
    logger.info("Cleaning up memory...")
    del state_dict
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Memory cleanup complete")
    
    return 0  # No VRAM tracking for CPU-only mode


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for converter.py
    If run without arguments, tries to launch app.py if available.
    Otherwise shows help information.
    """
    if len(sys.argv) == 1:
        # No arguments provided - try to launch app.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "app.py")
        
        if os.path.exists(app_path):
            print("Launching interactive mode (app.py)...\n")
            try:
                import subprocess
                result = subprocess.run([sys.executable, app_path], check=False)
                sys.exit(result.returncode)
            except Exception as e:
                print(f"Error launching app.py: {e}")
                print("\nTry running: python app.py")
                sys.exit(1)
        else:
            # app.py not found, show help
            print("=" * 70)
            print("FLUX MODEL PRECISION CONVERTER - Core Module")
            print("=" * 70)
            print("\nThis is the core converter module.")
            print("For interactive mode, use: app.py")
            print("\nAvailable conversion function:")
            print("  convert_precision(input_model, output_model, save_precision, ...)")
            print("\nUsage example in Python:")
            print("  from converter import convert_precision")
            print("  convert_precision('model.safetensors', 'model_fp16.safetensors', 'fp16')")
            print("\nFor CLI mode with arguments, use app.py")
            print("=" * 70)
            sys.exit(0)
    else:
        # Arguments provided but this is the core module - show guidance
        print("This is the core converter module, not meant to be run with CLI arguments.")
        print("Use app.py for CLI mode:")
        print("  python app.py --input_model <file> --output_model <file> --save_precision <precision>")
        sys.exit(1)


if __name__ == "__main__":
    main()
