from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create RGB-T manifest JSONL")
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="train", help="Split name")
    parser.add_argument("--rgb-dir", type=str, default="rgb", help="RGB directory name")
    parser.add_argument("--thermal-dir", type=str, default="thermal", help="Thermal directory name")
    parser.add_argument("--mask-dir", type=str, default="masks", help="Segmentation mask directory name")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--classification", action="store_true", help="Emit classification entries")
    parser.add_argument("--default-label", type=int, default=0, help="Default class label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rgb_dir = root / args.rgb_dir / args.split
    thermal_dir = root / args.thermal_dir / args.split
    mask_dir = root / args.mask_dir / args.split

    thermal_index = {
        p.stem: p
        for p in thermal_dir.glob("**/*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w") as f:
        for rgb_path in sorted([p for p in rgb_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]):
            stem = rgb_path.stem
            thermal_path = thermal_index.get(stem)
            if thermal_path is None:
                continue

            item = {
                "rgb": str(rgb_path.relative_to(root)),
                "thermal": str(thermal_path.relative_to(root)),
            }

            if args.classification:
                item["label"] = args.default_label
            else:
                mask_path = mask_dir / f"{stem}.png"
                if not mask_path.exists():
                    continue
                item["mask"] = str(mask_path.relative_to(root))

            f.write(json.dumps(item) + "\n")
            count += 1

    print(f"Wrote {count} samples to {output}")


if __name__ == "__main__":
    main()
