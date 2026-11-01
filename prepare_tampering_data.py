import json
import random
import shutil
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path("dataset_download")
OUTPUT = Path("data/tampering")
SEED = 42

random.seed(SEED)

# Locate the dataset's annotation and image folders.
annotation_dir = ROOT / "annotations"
if not annotation_dir.exists():
    raise FileNotFoundError(f"Annotations folder not found: {annotation_dir}")

image_files = {}
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image_files[path.stem] = path

records = []
skipped = []

for annotation_path in annotation_dir.glob("*.json"):
    try:
        annotation = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        skipped.append((annotation_path.name, str(exc)))
        continue

    authentic = annotation.get("is_authentic")

    if not isinstance(authentic, bool):
        skipped.append((annotation_path.name, "Missing/invalid is_authentic"))
        continue

    source_id = str(annotation.get("id") or annotation_path.stem)

    # Match image variants such as C0, C2 and C4 to this source document.
    variants = [
        (stem, path)
        for stem, path in image_files.items()
        if stem == source_id
        or stem.startswith(source_id + "_")
    ]

    if not variants:
        skipped.append((annotation_path.name, "No matching image"))
        continue

    label = "genuine" if authentic else "tampered"

    for stem, image_path in variants:
        records.append({
            "path": image_path,
            "label": label,
            "group": source_id,
        })

if not records:
    raise RuntimeError(
        "No records found. Check annotation IDs against image filenames."
    )

# Split by source document, never by individual image.
groups = sorted({r["group"] for r in records})
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
train_idx, test_idx = next(
    splitter.split(groups, groups=groups)
)
train_groups = {groups[i] for i in train_idx}
test_groups = {groups[i] for i in test_idx}

# Verify both classes occur in both partitions before copying.
for split_name, selected_groups in [
    ("train", train_groups),
    ("test", test_groups),
]:
    labels = {r["label"] for r in records if r["group"] in selected_groups}
    if labels != {"genuine", "tampered"}:
        raise RuntimeError(
            f"{split_name} split does not contain both classes: {labels}. "
            "Use a stratified group split before proceeding."
        )

for split_name, selected_groups in [
    ("train", train_groups),
    ("test", test_groups),
]:
    for label in ("genuine", "tampered"):
        destination = OUTPUT / split_name / label
        destination.mkdir(parents=True, exist_ok=True)

        for record in records:
            if record["group"] not in selected_groups:
                continue
            if record["label"] != label:
                continue

            # Keep each filename unique in the destination.
            target = destination / (
                f"{record['group']}__{record['path'].name}"
            )
            if not target.exists():
                shutil.copy2(record["path"], target)

print("Dataset preparation finished.")
print("Source documents:", len(groups))
print("Matched image variants:", len(records))
print("Train source documents:", len(train_groups))
print("Test source documents:", len(test_groups))
print("Skipped annotations:", len(skipped))
for split_name in ("train", "test"):
    for label in ("genuine", "tampered"):
        folder = OUTPUT / split_name / label
        print(f"{split_name}/{label}: {len(list(folder.glob('*')))} images")

if skipped:
    print("First skipped entries:")
    for item in skipped[:10]:
        print(item)
        