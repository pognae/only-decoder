import json
from pathlib import Path

train_path = Path("train_data/train.jsonl")
silver_path = Path("train_data/silver_mixed_3000.jsonl")
out_path = Path("train_data/train_merged.jsonl")

seen = set()
kept_train = 0
dup_train = 0
added_silver = 0
dup_silver = 0
invalid_silver = 0

# 사용법
# wc -l train_data/train.jsonl train_data/silver_mixed_3000.jsonl train_data/train_merged.jsonl
# mv train_data/train_merged.jsonl train_data/train.jsonl



with out_path.open("w", encoding="utf-8") as out:
    # 1) 기존 train 우선 유지
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("record_id", ""))
            if not rid:
                continue
            if rid in seen:
                dup_train += 1
                continue
            seen.add(rid)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_train += 1

    # 2) silver는 중복 없는 것만 추가 (라벨 필수)
    with silver_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("record_id", ""))
            if not rid or rid in seen:
                dup_silver += 1
                continue
            if "command" not in row or "reason" not in row:
                invalid_silver += 1
                continue
            seen.add(rid)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            added_silver += 1

print("saved:", out_path)
print("kept_train:", kept_train, "dup_train:", dup_train)
print("added_silver:", added_silver, "dup_silver:", dup_silver, "invalid_silver:", invalid_silver)
print("total_out:", kept_train + added_silver)
