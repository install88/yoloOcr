"""
Step 1: Analyze the PaddleOCR-format dataset.
Outputs:
  - Total image count, date-region count
  - Sample of date transcriptions
  - Distribution of date formats
"""
import json
import re
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("C:/Users/insta/Desktop/dataset")
TRAIN_LABEL = DATASET_DIR / "train_label.txt"
VAL_LABEL   = DATASET_DIR / "val_label.txt"

DATE_PATTERNS = [
    (r"\d{4}[./\-]\d{1,2}[./\-]\d{1,2}", "YYYY.MM.DD"),
    (r"\d{1,2}[./\-]\d{1,2}[./\-]\d{4}", "DD.MM.YYYY"),
    (r"(有效|製造|到期|EXP|MFG|MFD|BB)[^\d]*\d{4,8}", "中文/英文前綴+數字"),
    (r"\d{6}", "YYMMDD 六位"),
]

def classify_date_format(text: str) -> str:
    for pattern, name in DATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return "其他"

def analyze_label_file(label_path: Path):
    total_images = 0
    total_regions = 0
    date_regions = 0
    format_counter: Counter = Counter()
    samples: list[str] = []

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_images += 1
            tab_idx = line.index("\t")
            annotations = json.loads(line[tab_idx + 1:])
            for ann in annotations:
                total_regions += 1
                text = ann.get("transcription", "")
                if text == "###":
                    continue
                fmt = classify_date_format(text)
                if fmt != "其他":
                    date_regions += 1
                    format_counter[fmt] += 1
                    if len(samples) < 8:
                        samples.append(text)

    return {
        "images": total_images,
        "total_regions": total_regions,
        "date_regions": date_regions,
        "format_counter": format_counter,
        "samples": samples,
    }

def main():
    print("=" * 55)
    print("  Dataset Analysis")
    print("=" * 55)

    for split, path in [("Train", TRAIN_LABEL), ("Val", VAL_LABEL)]:
        r = analyze_label_file(path)
        print(f"\n[{split}]")
        print(f"  Images         : {r['images']}")
        print(f"  Total regions  : {r['total_regions']}")
        print(f"  Date regions   : {r['date_regions']}")
        print(f"  Format breakdown:")
        for fmt, cnt in r["format_counter"].most_common():
            print(f"    {fmt:<25} {cnt}")
        print(f"  Sample transcriptions:")
        for s in r["samples"]:
            print(f"    {s}")

    print("\nDone.")

if __name__ == "__main__":
    main()
