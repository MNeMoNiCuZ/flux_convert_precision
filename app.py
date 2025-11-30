#!/usr/bin/env python3
"""
Flux Model Precision Converter - CLI Application

This is the command-line interface wrapper for the Flux model precision converter.

Supports two modes:
1. Interactive Mode (no arguments): Guided prompts for batch conversion
2. CLI Mode (with arguments): Command-line interface for automation

Interactive Mode Usage:
    python app.py
    (Then follow the prompts to select files and precision)

CLI Mode Usage:
    python app.py --input_model path/to/model.safetensors --output_model path/to/output.safetensors --save_precision fp16

Requirements:
    - torch
    - safetensors
    - tqdm
"""

import argparse
import os
import sys
from typing import List

import torch
from safetensors import safe_open

# Import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Graceful degradation if colorama is not available
    COLORAMA_AVAILABLE = False
    class Fore:
        GREEN = ''
        YELLOW = ''
        RESET = ''
    class Style:
        RESET_ALL = ''

# Import from the converter module
from converter import (
    logger,
    normalize_path,
    detect_precision_from_filename,
    generate_output_filename,
    convert_precision,
    str_to_dtype,
)


# ============================================================================
# Interactive Mode Functions
# ============================================================================

def get_input_files() -> List[str]:
    """
    Prompt user for input file paths.
    Supports multiple files (one per line).
    
    Returns:
        List of normalized file paths
    """
    print("\n" + "="*70)
    print("FLUX MODEL PRECISION CONVERTER - Interactive Mode")
    print("="*70)
    print("\nInput path(s) (absolute or relative):")
    
    paths = []
    while True:
        line = input().strip()
        if not line:
            if paths:
                break
            else:
                print("Please enter at least one file path.")
                continue
        
        # Normalize and validate path
        normalized = normalize_path(line)
        
        # If file doesn't exist and has no extension, try adding .safetensors
        if not os.path.isfile(normalized):
            # Check if the input has no extension
            _, ext = os.path.splitext(normalized)
            if not ext:
                # Try adding .safetensors
                safetensors_path = normalized + ".safetensors"
                if os.path.isfile(safetensors_path):
                    normalized = safetensors_path
                    print(f"Auto-detected: {normalized}")
                else:
                    print(f"{Fore.RED}File not found: {normalized} or {safetensors_path}{Style.RESET_ALL}")
                    response = input("Continue anyway? (y/n): ").strip().lower()
                    if response != 'y':
                        continue
            else:
                print(f"{Fore.RED}File not found: {normalized}{Style.RESET_ALL}")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    continue
        
        
        paths.append(normalized)
        print(f"Added: {normalized}")
        
        if len(paths) == 1:
            print("\nPress Enter again to finish, or paste another path:")
    
    return paths


def get_precision_choice(input_files: List[str] = None, file_precisions: dict = None) -> List[str]:
    """
    Display precision menu as a table and get user choice(s).
    Supports multiple selections.
    
    Args:
        input_files: List of input file paths (numbered 1, 2, 3...)
        file_precisions: Dictionary mapping file paths to detected precision strings
    
    Returns:
        List of selected precision strings
    """
    print("\n\n" + "="*70)
    print("STEP 2: SELECT OUTPUT PRECISION(S)")
    print("="*70)
    
    # Build a mapping of precision to file indices
    precision_to_indices = {
        'fp32': [],
        'fp16': [],
        'bf16': [],
        'fp8': []
    }
    
    if input_files and file_precisions:
        for idx, filepath in enumerate(input_files, 1):
            detected = file_precisions.get(filepath)
            if detected:
                # Normalize detected precision to our standard names
                detected_lower = detected.lower()
                if 'float32' in detected_lower or 'fp32' in detected_lower:
                    precision_to_indices['fp32'].append(idx)
                elif 'float16' in detected_lower or 'fp16' in detected_lower:
                    precision_to_indices['fp16'].append(idx)
                elif 'bfloat16' in detected_lower or 'bf16' in detected_lower:
                    precision_to_indices['bf16'].append(idx)
                elif 'float8' in detected_lower or 'fp8' in detected_lower:
                    precision_to_indices['fp8'].append(idx)
            
            # Also check for existing output files in the same folder
            directory = os.path.dirname(filepath)
            for target_prec in ['fp32', 'fp16', 'bf16', 'fp8']:
                # Generate the expected output filename
                expected_output = generate_output_filename(filepath, target_prec)
                if os.path.exists(expected_output) and expected_output != filepath:
                    # File exists in folder, add to indices if not already there
                    if idx not in precision_to_indices[target_prec]:
                        precision_to_indices[target_prec].append(idx)
    
    # Display table header
    print("\nAvailable precision formats:")
    print("-" * 85)
    print(f"{'#':<4} {'Precision':<12} {'Est. Size':<12} {'Status':<30} {'Input Models':<20}")
    print("-" * 85)
    
    # Precision options with their details
    # (Number, Name, Key, Est Size)
    precisions = [
        ('1', 'FP32', 'fp32', '44.3 GB'),
        ('2', 'FP16', 'fp16', '22 GB'),
        ('3', 'BF16', 'bf16', '22 GB'),
        ('4', 'FP8', 'fp8', '11 GB'),
    ]
    
    for num, name, key, size in precisions:
        indices = precision_to_indices[key]
        
        if indices:
            # This precision exists in input - mark as orange
            status = "Already in input"
            models_str = f"Models: {', '.join(map(str, indices))}"
            
            if COLORAMA_AVAILABLE:
                line = f"{Fore.YELLOW}{num:<4} {name:<12} {size:<12} {status:<30} {models_str}{Style.RESET_ALL}"
            else:
                line = f"{num:<4} {name:<12} {size:<12} {status:<30} {models_str}"
        else:
            # New precision - mark as green
            status = "Available"
            models_str = "-"
            
            if COLORAMA_AVAILABLE:
                line = f"{Fore.GREEN}{num:<4} {name:<12} {size:<12} {status:<30} {models_str}{Style.RESET_ALL}"
            else:
                line = f"{num:<4} {name:<12} {size:<12} {status:<30} {models_str}"
        
        print(line)
    
    print("-" * 85)
    
    # Legend
    if COLORAMA_AVAILABLE:
        print(f"\nLegend:")
        print(f"  {Fore.GREEN}Green{Style.RESET_ALL}  = New precision (will convert)")
        print(f"  {Fore.YELLOW}Orange{Style.RESET_ALL} = Existing precision (will skip if selected)")
    else:
        print(f"\nNote: Precisions marked 'Already in input' will be skipped if selected")
    
    print("\nYou can select multiple formats (comma-separated or not, e.g., '2,4' or '24')")
    print("\nEnter your choice(s) (1-4): ")
    
    precision_map = {
        '1': 'fp32',
        '2': 'fp16',
        '3': 'bf16',
        '4': 'fp8',
    }
    
    while True:
        choice = input().strip()
        
        # Parse input - support both "1,2,4" and "124" formats
        if ',' in choice:
            # Comma-separated format
            choices = [c.strip() for c in choice.split(',')]
        else:
            # Concatenated format (e.g., "124" -\u003e ["1", "2", "4"])
            # Split each character but filter out spaces
            choices = [c for c in choice if c.strip() and c.isdigit()]
        
        selected_precisions = []
        invalid = False
        
        for c in choices:
            if c in precision_map:
                prec = precision_map[c]
                if prec not in selected_precisions:
                    selected_precisions.append(prec)
            else:
                print(f"Invalid choice: '{c}'. Please enter numbers 1-4.")
                invalid = True
                break
        
        if not invalid and selected_precisions:
            print(f"Selected: {', '.join([p.upper() for p in selected_precisions])}")
            return selected_precisions
        elif not invalid:
            print("Please enter at least one valid choice.")


def ask_input_precision_for_file(filepath: str) -> str:
    """
    Ask user for the input precision of a file when it can't be detected.
    
    Args:
        filepath: Path to the file
    
    Returns:
        User-specified input precision string
    """
    print(f"\n[WARNING] Cannot detect input precision from filename: {os.path.basename(filepath)}")
    print("Please specify the current precision:")
    print("  1. FP32")
    print("  2. FP16")
    print("  3. BF16")
    print("  4. FP8")
    print("\nEnter choice (1-4): ")
    
    precision_map = {
        '1': 'fp32',
        '2': 'fp16',
        '3': 'bf16',
        '4': 'fp8',
    }
    
    while True:
        choice = input().strip()
        if choice in precision_map:
            return precision_map[choice]
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4:")



# ============================================================================
# CLI Interface
# ============================================================================

def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Convert Flux models between different precisions (CPU-only mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported precisions:
  - float, float32, fp32  : 32-bit floating point
  - fp16, float16         : 16-bit floating point
  - bf16, bfloat16        : Brain float 16
  - fp8, float8           : 8-bit floating point (default: fp8_e4m3fn)
  - fp8_e4m3fn            : 8-bit floating point E4M3
  - fp8_e5m2              : 8-bit floating point E5M2
        """
    )

    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to input Flux model (safetensors file)"
    )

    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Path to save the converted model (safetensors file)"
    )

    parser.add_argument(
        "--save_precision",
        type=str,
        required=True,
        choices=["float", "float32", "fp32", "fp16", "float16", "bf16", "bfloat16", "fp8", "float8", 
                 "fp8_e4m3fn", "fp8_e5m2", "fp8_e4m3fnuz", "fp8_e5m2fnuz"],
        help="Target precision for the output model"
    )

    parser.add_argument(
        "--mem_eff",
        action="store_true",
        help="Use memory efficient loading and saving (recommended for large models)"
    )

    return parser


def run_cli_mode(parser: argparse.ArgumentParser):
    """Run in CLI mode with command-line arguments"""
    args = parser.parse_args()
    
    # Normalize paths
    args.input_model = normalize_path(args.input_model)
    args.output_model = normalize_path(args.output_model)

    # Validate input file exists, try adding .safetensors if needed
    if not os.path.isfile(args.input_model):
        _, ext = os.path.splitext(args.input_model)
        if not ext:
            # Try adding .safetensors
            safetensors_path = args.input_model + ".safetensors"
            if os.path.isfile(safetensors_path):
                args.input_model = safetensors_path
                logger.info(f"Auto-detected input file with .safetensors extension: {args.input_model}")
            else:
                logger.error(f"Input model not found: {args.input_model} or {safetensors_path}")
                sys.exit(1)
        else:
            logger.error(f"Input model not found: {args.input_model}")
            sys.exit(1)


    # Create output directory if needed
    output_dir = os.path.dirname(args.output_model)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Force CPU-only mode
    loading_device = "cpu"
    working_device = "cpu"

    # Run conversion
    try:
        convert_precision(
            input_model=args.input_model,
            output_model=args.output_model,
            save_precision=args.save_precision,
            loading_device=loading_device,
            working_device=working_device,
            mem_eff_load_save=args.mem_eff,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


def run_interactive_mode():
    """Run in interactive mode with user prompts"""
    try:
        # Get input files from user
        input_files = get_input_files()
        
        if not input_files:
            print("\n[ERROR] No input files provided. Exiting.")
            sys.exit(0)
        
        # STEP 1: Detect input precision for all files
        print("\n" + "="*70)
        print("STEP 1: DETECTING INPUT PRECISIONS")
        print("="*70)
        
        file_precisions = {}
        file_casing = {}  # Store whether input used uppercase (True) or lowercase (False)
        
        for idx, input_file in enumerate(input_files, 1):
            print(f"\n{idx}. {os.path.basename(input_file)}")
            
            # Detect casing from filename
            # Strategy: Find the precision string in the filename and check its casing
            filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(filename)[0]
            detected_prec_from_name = detect_precision_from_filename(filename)
            
            if detected_prec_from_name:
                # Find the actual occurrence in the filename to detect casing
                import re
                # Use lookarounds to match precision not surrounded by alphanumeric chars
                # This allows matching _FP16 (where \b would fail because _ is a word char)
                pattern = re.compile(rf'(?<![a-zA-Z0-9]){re.escape(detected_prec_from_name)}(?![a-zA-Z0-9])', re.IGNORECASE)
                match = pattern.search(name_without_ext)  # Search in name without extension
                if match:
                    actual_in_file = match.group(0)
                    file_casing[input_file] = actual_in_file.isupper()
                else:
                    file_casing[input_file] = False
            else:
                file_casing[input_file] = False
            
            try:
                with safe_open(input_file, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    if keys:
                        sample_tensor = f.get_tensor(keys[0])
                        detected_precision = str(sample_tensor.dtype).replace('torch.', '')
                        print(f"   Input precision: {detected_precision}")
                        file_precisions[input_file] = detected_precision
            except Exception as e:
                print(f"   {Fore.RED}Could not detect: {e}{Style.RESET_ALL}")
                file_precisions[input_file] = None
        
        # For multiple files, ask if user wants batch or individual precision selection
        file_target_precisions = {}  # file -> list of target precisions
        
        if len(input_files) > 1:
            print("\n\n" + "="*70)
            print("STEP 2: PRECISION SELECTION MODE")
            print("="*70)
            print(f"\nYou have {len(input_files)} input files.")
            print("1. Select output precision(s) once for all files [DEFAULT]")
            print("2. Select output precision(s) individually for each file")
            print("\nChoice (1-2) [default: 1]: ")
            
            mode_choice = input().strip()
            individual_mode = (mode_choice == '2')
        else:
            individual_mode = False
        
        if individual_mode:
            # Individual selection for each file
            for idx, input_file in enumerate(input_files, 1):
                print(f"\n{'-'*70}")
                print(f"File {idx}: {os.path.basename(input_file)}")
                if file_precisions.get(input_file):
                    print(f"Input precision: {file_precisions[input_file]}")
                print(f"{'-'*70}")
                
                # Pass input files and file_precisions for individual mode
                # For individual mode, pass just the single file
                target_precs = get_precision_choice([input_file], file_precisions)
                file_target_precisions[input_file] = target_precs
        else:
            # Batch selection for all files
            # Pass all input files and file_precisions
            target_precisions = get_precision_choice(input_files, file_precisions)
            
            # Apply same target precisions to all files
            for input_file in input_files:
                file_target_precisions[input_file] = target_precisions
        
        # STEP 3: Device selection
        print("\n\n" + "="*70)
        # STEP 3: DEVICE SELECTION (Skipped - CPU Only)
        loading_device = "cpu"
        working_device = "cpu"
        
        # STEP 4: Begin conversion
        total_files = len(input_files)
        total_target_conversions = sum(len(file_target_precisions[f]) for f in input_files)
        
        print("\n\n" + "="*70)
        print(f"STEP 4: PROCESSING {total_files} FILE(S) -> {total_target_conversions} TARGET CONVERSION(S)")
        print("="*70)
        
        total_conversions = 0
        skipped_conversions = 0
        conversion_results = []  # Track results: (input_file, target_prec, output_file, success, file_size_gb, peak_vram_gb)
        
        for file_idx, input_file in enumerate(input_files, 1):
            print(f"\n{'='*70}")
            print(f"FILE {file_idx} of {total_files}")
            print(f"{'='*70}")
            print(f"Input: {input_file}")
            
            # Get the detected precision from earlier
            detected_input_precision = file_precisions.get(input_file)
            if detected_input_precision:
                print(f"Input precision: {detected_input_precision}")
            
            # Auto-enable memory-efficient mode based on file size
            # Don't ask user, just auto-detect
            mem_eff = False
            try:
                file_size_gb = os.path.getsize(input_file) / (1024**3)
                # Auto-enable for models over 20GB
                if file_size_gb > 20:
                    mem_eff = True
                    print(f"Large file ({file_size_gb:.1f} GB) - auto-enabled memory-efficient mode")
            except Exception:
                pass
            
            # Process each target precision for this file
            target_precisions = file_target_precisions[input_file]
            
            for target_precision in target_precisions:
                # Check if output precision matches input precision
                target_dtype_str = str(str_to_dtype(target_precision)).replace('torch.', '')
                if detected_input_precision and target_dtype_str == detected_input_precision:
                    print(f"\nSkipping {target_precision.upper()} (already in this precision)")
                    skipped_conversions += 1
                    continue
                
                # Detect precision from filename for naming purposes
                detected_filename_precision = detect_precision_from_filename(input_file)
                use_uppercase = file_casing.get(input_file, False)
                
                if not detected_filename_precision:
                    # Ask user for input precision (for filename generation only)
                    if len(target_precisions) > 1:
                        print(f"\nFor {target_precision.upper()} output: Input precision needed for filename")
                    else:
                        print("\nInput precision needed for generating output filename")
                    user_specified_precision = ask_input_precision_for_file(input_file)
                    # Temporarily modify filename for output generation
                    basename = os.path.basename(input_file)
                    name_without_ext = os.path.splitext(basename)[0]
                    temp_filename = f"{name_without_ext}_{user_specified_precision}.safetensors"
                    temp_path = os.path.join(os.path.dirname(input_file), temp_filename)
                    output_file = generate_output_filename(temp_path, target_precision, use_uppercase)
                else:
                    output_file = generate_output_filename(input_file, target_precision, use_uppercase)
                
                print(f"\n-> Converting to {target_precision.upper()}")
                print(f"  Output: {output_file}")
                
                # Check if output file already exists
                if os.path.exists(output_file):
                    print(f"  {Fore.YELLOW}Output file already exists!{Style.RESET_ALL}")
                    response = input("  Overwrite? (y/n): ").strip().lower()
                    if response != 'y':
                        print("  Skipping this conversion.")
                        skipped_conversions += 1
                        continue
                
                # Determine devices for this specific conversion
                current_loading_device = loading_device
                current_working_device = working_device
                current_mem_eff = mem_eff
                
                # Run conversion
                try:
                    peak_vram_gb = convert_precision(
                        input_model=input_file,
                        output_model=output_file,
                        save_precision=target_precision,
                        loading_device=current_loading_device,
                        working_device=current_working_device,
                        mem_eff_load_save=current_mem_eff,
                    )
                    
                    # Get output file size
                    output_size_gb = 0
                    if os.path.exists(output_file):
                        output_size_gb = os.path.getsize(output_file) / (1024**3)
                    
                    print(f"  Successfully converted to {target_precision.upper()}")
                    if peak_vram_gb > 0:
                        print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
                    print(f"  Output size: {output_size_gb:.2f} GB")
                    
                    # Track successful conversion
                    conversion_results.append((
                        os.path.basename(input_file),
                        target_precision.upper(),
                        os.path.basename(output_file),
                        True,
                        output_size_gb,
                        peak_vram_gb
                    ))
                    
                    total_conversions += 1
                except Exception as e:
                    print(f"  {Fore.RED}Conversion failed: {e}{Style.RESET_ALL}")
                    logger.error(f"Conversion failed for {input_file} -> {target_precision}: {e}", exc_info=True)
                    
                    # Track failed conversion
                    conversion_results.append((
                        os.path.basename(input_file),
                        target_precision.upper(),
                        os.path.basename(output_file),
                        False,
                        0,
                        0
                    ))
                    
                    continue_response = input("\n  Continue with remaining conversions? (Y/n): ").strip().lower()
                    if continue_response == 'n':
                        print(f"\n{Fore.RED}Stopping batch conversion.{Style.RESET_ALL}")
                        sys.exit(1)
        
        print("\n\n" + "="*70)
        print("BATCH CONVERSION COMPLETE")
        print("="*70)
        
        # Display results summary table
        if conversion_results:
            print("\n" + "-"*60)
            print("CONVERSION RESULTS SUMMARY")
            print("-"*60)
            print(f"{'Output Precision':<18} {'File Size':<12} {'Status':<15}")
            print("-"*60)
            
            for input_name, target_prec, output_name, success, file_size_gb, peak_vram_gb in conversion_results:
                status = "Success" if success else "Failed"
                file_size_str = f"{file_size_gb:.2f} GB" if file_size_gb > 0 else "-"
                
                print(f"{target_prec:<18} {file_size_str:<12} {status:<15}")
            
            print("-"*60)
        
        print(f"\nSuccessful conversions: {total_conversions}")
        if skipped_conversions > 0:
            print(f"Skipped: {skipped_conversions}")
        print("All files saved to their original directories.")
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point - supports both CLI mode and interactive mode"""
    parser = setup_parser()
    
    # Check if running in interactive mode (no command-line arguments)
    if len(sys.argv) == 1:
        # Interactive mode
        run_interactive_mode()
    else:
        # CLI mode (original behavior)
        run_cli_mode(parser)


if __name__ == "__main__":
    main()
