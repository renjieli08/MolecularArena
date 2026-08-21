import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update an older Chemformer checkpoint to the newer hyperparameter key layout."
    )
    parser.add_argument("--input-ckpt", type=Path, required=True, help="Path to the original .ckpt file")
    parser.add_argument(
        "--output-ckpt",
        type=Path,
        default=None,
        help="Path to save the updated checkpoint. Defaults to <input_stem>_v2.ckpt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_ckpt = args.input_ckpt.resolve()
    output_ckpt = args.output_ckpt.resolve() if args.output_ckpt else input_ckpt.with_name(f"{input_ckpt.stem}_v2.ckpt")

    checkpoint = torch.load(input_ckpt, map_location="cpu")
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        raise KeyError("Checkpoint does not contain a 'hyper_parameters' section.")

    if "vocabulary_size" in hyper_parameters:
        print(f"'vocabulary_size' already exists in {input_ckpt}.")
    elif "vocab_size" in hyper_parameters:
        hyper_parameters["vocabulary_size"] = hyper_parameters.pop("vocab_size")
        print("Renamed hyperparameter key: vocab_size -> vocabulary_size")
    else:
        raise KeyError("Neither 'vocab_size' nor 'vocabulary_size' was found in checkpoint hyper_parameters.")

    torch.save(checkpoint, output_ckpt)
    print(f"Saved updated checkpoint to: {output_ckpt}")


if __name__ == "__main__":
    main()
