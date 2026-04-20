import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sllm.common.experiments import DEFAULT_RUN_ID, build_run_paths, experiments_root_from_env
from sllm.common.io import write_json


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _resolve_source_dirs(
    run_id: str,
    *,
    experiments_root: str,
    default_model_dir: str,
    default_tokenizer_dir: str,
) -> Dict[str, str]:
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if rid == DEFAULT_RUN_ID:
        return {
            "run_id": rid,
            "model_dir": os.path.abspath(default_model_dir),
            "tokenizer_dir": os.path.abspath(default_tokenizer_dir),
        }
    rp = build_run_paths(rid, root_dir=experiments_root)
    return {
        "run_id": rp.run_id,
        "model_dir": os.path.abspath(rp.model_dir),
        "tokenizer_dir": os.path.abspath(rp.tokenizer_dir),
    }


def _validate_source(
    *,
    model_dir: str,
    min_command_accuracy: Optional[float],
    min_reason_coverage: float,
    max_command_malformed_ratio: float,
    require_review_approval: bool,
) -> Dict[str, Any]:
    errors = []
    checks: Dict[str, Any] = {}

    required_files = [
        "training_result.json",
        "metrics.json",
        "data_quality_report.json",
        "reason_coverage_report.json",
        "config.json",
        "tokenizer.json",
    ]
    missing_files = [name for name in required_files if not os.path.exists(os.path.join(model_dir, name))]
    if missing_files:
        errors.append(f"required artifacts missing: {missing_files}")
    checks["required_files"] = {
        "expected": required_files,
        "missing": missing_files,
    }

    quality = _read_json(os.path.join(model_dir, "data_quality_report.json")) or {}
    quality_status = quality.get("status")
    checks["quality_status"] = quality_status
    if quality_status != "PASS":
        errors.append(f"data_quality_report.status must be PASS (current={quality_status})")

    metrics = _read_json(os.path.join(model_dir, "metrics.json")) or {}
    command_accuracy = _safe_float(metrics.get("command_accuracy"))
    checks["command_accuracy"] = command_accuracy
    if min_command_accuracy is not None:
        if command_accuracy is None or command_accuracy < min_command_accuracy:
            errors.append(
                f"command_accuracy {command_accuracy} is lower than required {min_command_accuracy}"
            )

    reason_cov = _read_json(os.path.join(model_dir, "reason_coverage_report.json")) or {}
    rc = reason_cov.get("reason_coverage") or {}
    cs = reason_cov.get("command_shape") or {}
    coverage_ratio = _safe_float(rc.get("coverage_ratio"))
    malformed_ratio = _safe_float(cs.get("malformed_ratio"))
    checks["reason_coverage_ratio"] = coverage_ratio
    checks["command_malformed_ratio"] = malformed_ratio
    if coverage_ratio is None or coverage_ratio < float(min_reason_coverage):
        errors.append(
            f"reason coverage ratio {coverage_ratio} is lower than required {min_reason_coverage}"
        )
    if malformed_ratio is None or malformed_ratio > float(max_command_malformed_ratio):
        errors.append(
            f"command malformed ratio {malformed_ratio} exceeds allowed {max_command_malformed_ratio}"
        )

    review = _read_json(os.path.join(model_dir, "review_approval.json")) or {}
    review_approved = bool(review.get("approved"))
    checks["review_approved"] = review_approved
    checks["review_reviewer"] = review.get("reviewer")
    checks["review_approved_at_utc"] = review.get("approved_at_utc")
    if require_review_approval and not review_approved:
        errors.append("review approval required: review_approval.json with approved=true")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "checks": checks,
    }


def _copy_dir(src: str, dst: str) -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _promote_dir(
    *,
    src_dir: str,
    target_dir: str,
    backup_root: str,
) -> Dict[str, Any]:
    src_abs = os.path.abspath(src_dir)
    target_abs = os.path.abspath(target_dir)
    if src_abs == target_abs:
        return {"skipped": True, "reason": "source_dir and target_dir are identical"}

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    os.makedirs(backup_root, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = None
    if os.path.exists(target_dir):
        base = os.path.basename(target_dir.rstrip(os.sep)) or "target"
        backup_path = os.path.join(backup_root, f"{base}_{timestamp}")
        shutil.move(target_dir, backup_path)

    temp_path = os.path.join(os.path.dirname(target_dir), f".promote_tmp_{timestamp}")
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    shutil.copytree(src_dir, temp_path)
    shutil.move(temp_path, target_dir)
    return {"backup_path": backup_path}


def _merge_dir_contents(src_dir: str, dst_dir: str) -> Dict[str, Any]:
    """
    Merge src directory contents into dst without deleting dst.
    Used when tokenizer/model share the same target directory.
    """
    copied_files = 0
    copied_dirs = 0
    if not os.path.isdir(src_dir):
        return {"copied_files": 0, "copied_dirs": 0, "reason": "source_not_dir"}
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied_dirs += 1
        else:
            shutil.copy2(src, dst)
            copied_files += 1
    return {"copied_files": copied_files, "copied_dirs": copied_dirs}


def finalize_run(
    *,
    run_id: str,
    target_model_dir: str,
    target_tokenizer_dir: Optional[str],
    experiments_root: str,
    default_model_dir: str,
    default_tokenizer_dir: str,
    min_command_accuracy: Optional[float],
    min_reason_coverage: float,
    max_command_malformed_ratio: float,
    require_review_approval: bool,
    dry_run: bool = False,
) -> Dict[str, Any]:
    src = _resolve_source_dirs(
        run_id,
        experiments_root=experiments_root,
        default_model_dir=default_model_dir,
        default_tokenizer_dir=default_tokenizer_dir,
    )
    source_model_dir = src["model_dir"]
    source_tokenizer_dir = src["tokenizer_dir"]
    rid = src["run_id"]

    target_model_dir = os.path.abspath(target_model_dir)
    target_tokenizer_dir = os.path.abspath(target_tokenizer_dir or target_model_dir)
    releases_dir = os.path.abspath(os.path.join(os.path.dirname(target_model_dir), "releases"))
    backup_root = os.path.join(releases_dir, "backups")

    if not os.path.isdir(source_model_dir):
        return {
            "ok": False,
            "run_id": rid,
            "error": f"source model_dir not found: {source_model_dir}",
        }

    validation = _validate_source(
        model_dir=source_model_dir,
        min_command_accuracy=min_command_accuracy,
        min_reason_coverage=min_reason_coverage,
        max_command_malformed_ratio=max_command_malformed_ratio,
        require_review_approval=require_review_approval,
    )
    if not validation["ok"]:
        return {
            "ok": False,
            "run_id": rid,
            "source_model_dir": source_model_dir,
            "target_model_dir": target_model_dir,
            "validation": validation,
        }

    result = {
        "ok": True,
        "run_id": rid,
        "source_model_dir": source_model_dir,
        "source_tokenizer_dir": source_tokenizer_dir,
        "target_model_dir": target_model_dir,
        "target_tokenizer_dir": target_tokenizer_dir,
        "releases_dir": releases_dir,
        "validation": validation,
        "require_review_approval": bool(require_review_approval),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        return result

    promote_model = _promote_dir(
        src_dir=source_model_dir,
        target_dir=target_model_dir,
        backup_root=backup_root,
    )
    result["model_promote"] = promote_model

    source_tok_abs = os.path.abspath(source_tokenizer_dir)
    source_model_abs = os.path.abspath(source_model_dir)
    target_model_abs = os.path.abspath(target_model_dir)
    target_tok_abs = os.path.abspath(target_tokenizer_dir)
    if source_tok_abs == source_model_abs:
        result["tokenizer_promote"] = {"skipped": True, "reason": "source tokenizer_dir == source model_dir"}
    elif target_tok_abs == target_model_abs:
        # Prevent destructive overwrite when model/tokenizer share one LIVE directory.
        merged = _merge_dir_contents(source_tokenizer_dir, target_model_dir)
        result["tokenizer_promote"] = {
            "merged_into_model_dir": True,
            "target_dir": target_model_dir,
            **merged,
        }
    else:
        promote_tok = _promote_dir(
            src_dir=source_tokenizer_dir,
            target_dir=target_tokenizer_dir,
            backup_root=backup_root,
        )
        result["tokenizer_promote"] = promote_tok

    record = {
        "run_id": rid,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_dir": source_model_dir,
        "source_tokenizer_dir": source_tokenizer_dir,
        "target_model_dir": target_model_dir,
        "target_tokenizer_dir": target_tokenizer_dir,
        "validation": validation,
    }
    write_json(os.path.join(target_model_dir, "promotion_record.json"), record)

    os.makedirs(releases_dir, exist_ok=True)
    history_path = os.path.join(releases_dir, "promotion_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(releases_dir, "current_run_id.txt"), "w", encoding="utf-8") as f:
        f.write(rid + "\n")

    result["promotion_record"] = os.path.join(target_model_dir, "promotion_record.json")
    result["promotion_history"] = history_path
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="experiment run id to promote")
    ap.add_argument("--target_model_dir", default="artifacts/model_dev")
    ap.add_argument("--target_tokenizer_dir", default=None)
    ap.add_argument("--experiments_root", default=experiments_root_from_env())
    ap.add_argument("--default_model_dir", default="artifacts/model_dev")
    ap.add_argument("--default_tokenizer_dir", default="artifacts/model_dev")
    ap.add_argument("--min_command_accuracy", type=float, default=None)
    ap.add_argument("--min_reason_coverage", type=float, default=1.0)
    ap.add_argument("--max_command_malformed_ratio", type=float, default=0.0)
    ap.add_argument("--skip_review_approval", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    result = finalize_run(
        run_id=args.run_id,
        target_model_dir=args.target_model_dir,
        target_tokenizer_dir=args.target_tokenizer_dir,
        experiments_root=args.experiments_root,
        default_model_dir=args.default_model_dir,
        default_tokenizer_dir=args.default_tokenizer_dir,
        min_command_accuracy=args.min_command_accuracy,
        min_reason_coverage=args.min_reason_coverage,
        max_command_malformed_ratio=args.max_command_malformed_ratio,
        require_review_approval=not args.skip_review_approval,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
