from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitask_transformer_finetune import run_from_cli


if __name__ == "__main__":
    run_from_cli(
        default_model_name="ibm/MoLFormer-XL-both-10pct",
        default_output_dir=PROJECT_ROOT / "molformer" / "outputs",
        default_trust_remote_code=True,
    )
