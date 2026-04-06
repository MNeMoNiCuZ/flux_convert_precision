"""
ZiT Transformer Checkpoint Converter
====================================

Purpose
- Converts a single-file safetensors checkpoint that uses fused ZiT attention
  projections (`attention.qkv.weight`) into a checkpoint layout expected by
  ZiT transformer loaders that require split projections
  (`attention.to_q.weight`, `attention.to_k.weight`, `attention.to_v.weight`).
- Normalizes key names from common export layouts, including:
  - `model.diffusion_model.*` prefix removal
  - `attention.q_norm` -> `attention.norm_q`
  - `attention.k_norm` -> `attention.norm_k`
  - `attention.out` -> `attention.to_out.0`
  - `x_embedder.*` -> `all_x_embedder.2-1.*`
  - `final_layer.*` -> `all_final_layer.2-1.*`
- Always writes a matching single-file index JSON:
  `<output>.index.json`

Precision / dtype support
- Input precision is not restricted to BF16.
- The script preserves tensor dtypes from the source checkpoint (FP32, FP16,
  BF16, etc.) and does not force a cast.

How to run
1) Interactive mode (no arguments):
   - Prompts for source file path
   - Prompts for output filename
   - Writes output next to the source file
   Example:
     python zit_transformer_checkpoint_converter.py

2) CLI mode:
   Example:
     python zit_transformer_checkpoint_converter.py ^
       --src "E:\\path\\in.safetensors" ^
       --out "E:\\path\\out.safetensors"

Validation behavior
- If `--reference-index` exists (default:
  `diffusion_pytorch_model.safetensors.index.json`) and `--skip-compare` is
  not set, the script prints key overlap stats against that reference index.

Important scope note
- This converter is ZiT-structure specific. It is not a universal converter for
  arbitrary diffusion transformer architectures.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


QKV_SUFFIX = ".attention.qkv.weight"


def remap_key(key: str) -> str:
    if key.startswith("model.diffusion_model."):
        key = key[len("model.diffusion_model.") :]

    key = key.replace(".attention.q_norm.", ".attention.norm_q.")
    key = key.replace(".attention.k_norm.", ".attention.norm_k.")
    key = key.replace(".attention.out.", ".attention.to_out.0.")

    if key.startswith("x_embedder."):
        key = key.replace("x_embedder.", "all_x_embedder.2-1.", 1)

    if key.startswith("final_layer."):
        key = key.replace("final_layer.", "all_final_layer.2-1.", 1)

    return key


def convert_state_dict(src_path: Path) -> dict[str, torch.Tensor]:
    src_sd = load_file(str(src_path))
    out_sd: dict[str, torch.Tensor] = {}

    for src_key, tensor in src_sd.items():
        key = remap_key(src_key)

        if key.endswith(QKV_SUFFIX):
            # Convert fused qkv projection into separate q/k/v projections.
            base = key[: -len("qkv.weight")]
            q, k, v = torch.chunk(tensor, 3, dim=0)
            out_sd[f"{base}to_q.weight"] = q.contiguous()
            out_sd[f"{base}to_k.weight"] = k.contiguous()
            out_sd[f"{base}to_v.weight"] = v.contiguous()
            continue

        out_sd[key] = tensor

    return out_sd


def build_single_file_index(sd: dict[str, torch.Tensor], filename: str) -> dict:
    total_size = sum(t.numel() * t.element_size() for t in sd.values())
    weight_map = {k: filename for k in sd.keys()}
    return {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}


def compare_to_reference_index(converted_keys: set[str], reference_index: Path) -> None:
    obj = json.loads(reference_index.read_text(encoding="utf-8"))
    ref_keys = set(obj.get("weight_map", {}).keys())
    missing = sorted(ref_keys - converted_keys)
    extra = sorted(converted_keys - ref_keys)

    print(f"reference keys: {len(ref_keys)}")
    print(f"converted keys: {len(converted_keys)}")
    print(f"missing vs reference: {len(missing)}")
    print(f"extra vs reference: {len(extra)}")
    if missing:
        print("missing sample:", missing[:12])
    if extra:
        print("extra sample:", extra[:12])


def clean_input_path(raw: str) -> Path:
    # Accept quoted Windows/Linux paths and local relative paths.
    text = raw.strip().strip('"').strip("'")
    return Path(text).expanduser()


def ask_interactive_paths() -> tuple[Path, Path]:
    while True:
        raw_src = input("Source safetensors path: ").strip()
        src_path = clean_input_path(raw_src)
        if src_path.exists() and src_path.is_file():
            break
        print(f"File not found: {src_path}")

    default_name = f"{src_path.stem}_converted.safetensors"
    raw_out = input(f"Output file name [{default_name}]: ").strip()
    out_name = Path(raw_out.strip().strip('"').strip("'")).name if raw_out else default_name
    if not out_name.lower().endswith(".safetensors"):
        out_name = f"{out_name}.safetensors"

    out_path = src_path.parent / out_name
    return src_path, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fused-qkv ZiT transformer safetensors to ZiT-style split key names."
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Input safetensors file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output safetensors file.",
    )
    parser.add_argument(
        "--reference-index",
        default="diffusion_pytorch_model.safetensors.index.json",
        help="Optional reference index to compare key set against.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip comparison against reference index.",
    )
    args = parser.parse_args()

    interactive = len(sys.argv) == 1
    if interactive:
        src_path, out_path = ask_interactive_paths()
    else:
        if not args.src:
            raise ValueError("--src is required when using command-line arguments.")
        src_path = Path(args.src)
        out_path = Path(args.out) if args.out else src_path.with_name(f"{src_path.stem}_converted.safetensors")

    ref_path = Path(args.reference_index)

    if not src_path.exists():
        raise FileNotFoundError(f"Missing input file: {src_path}")

    converted = convert_state_dict(src_path)
    save_file(converted, str(out_path), metadata={"format": "pt"})
    print(f"saved: {out_path} ({len(converted)} tensors)")

    index_obj = build_single_file_index(converted, out_path.name)
    index_path = Path(str(out_path) + ".index.json")
    index_path.write_text(json.dumps(index_obj, indent=2), encoding="utf-8")
    print(f"saved index: {index_path}")

    if not args.skip_compare and ref_path.exists():
        compare_to_reference_index(set(converted.keys()), ref_path)


if __name__ == "__main__":
    main()
