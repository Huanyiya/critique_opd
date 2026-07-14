# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create small driver datasets for critique-conditioned ALFWorld OPD."""

import argparse
from pathlib import Path

from datasets import Dataset


def build_rows(size: int, split: str) -> list[dict]:
    """Create one row per ALFWorld environment trajectory to collect."""
    return [
        {
            "data_source": "alfworld",
            "agent_name": "alfworld_opd",
            # The environment replaces this placeholder after reset. It remains a
            # valid chat so RLHFDataset can perform its normal schema handling.
            "prompt": [{"role": "user", "content": "Start one text-only ALFWorld task."}],
            "ability": "embodied_text",
            "reward_model": {"style": "environment", "ground_truth": ""},
            "extra_info": {"index": index, "split": split},
        }
        for index in range(size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="~/data/alfworld_opd")
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument(
        "--eval-split",
        choices=("eval_in_distribution", "eval_out_of_distribution"),
        default="eval_in_distribution",
    )
    args = parser.parse_args()
    if args.train_size <= 0 or args.val_size <= 0:
        raise ValueError("--train-size and --val-size must both be positive.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(build_rows(args.train_size, "train")).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(build_rows(args.val_size, args.eval_split)).to_parquet(output_dir / "val.parquet")
    print(f"Wrote ALFWorld OPD driver data to {output_dir}")


if __name__ == "__main__":
    main()
