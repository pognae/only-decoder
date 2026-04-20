import argparse
import os
import shutil
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_rmtree(path: str, repo_root: str) -> bool:
    abspath = os.path.abspath(os.path.expanduser(path))
    # Only delete within this repo, and only under artifacts/
    if not abspath.startswith(repo_root + os.sep):
        raise ValueError(f"refuse to delete outside repo: {abspath}")
    rel = os.path.relpath(abspath, repo_root)
    if not (rel == "artifacts" or rel.startswith("artifacts" + os.sep)):
        raise ValueError(f"refuse to delete non-artifacts path: {abspath}")
    if not os.path.exists(abspath):
        return False
    shutil.rmtree(abspath)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="artifacts/model_dev")
    ap.add_argument("--tokenizer_dir", default="artifacts/tokenizer")
    ap.add_argument(
        "--delete_experiments",
        action="store_true",
        help="also delete artifacts/experiments/* (all runs)",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo_root = _repo_root()
    targets = [
        os.path.join(repo_root, args.model_dir),
        os.path.join(repo_root, args.tokenizer_dir),
    ]
    if args.delete_experiments:
        targets.append(os.path.join(repo_root, "artifacts", "experiments"))

    # De-dup and sort for stable output.
    targets = sorted(set(os.path.abspath(p) for p in targets))

    print("repo_root:", repo_root)
    print("dry_run:", bool(args.dry_run))
    print("targets:")
    for t in targets:
        print(" -", t)

    deleted = 0
    missing = 0
    for t in targets:
        if args.dry_run:
            if os.path.exists(t):
                deleted += 1
            else:
                missing += 1
            continue
        if _safe_rmtree(t, repo_root):
            deleted += 1
        else:
            missing += 1

    print("deleted:", deleted)
    print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

