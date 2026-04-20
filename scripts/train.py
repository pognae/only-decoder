import argparse
import json
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_import_path(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _run(cmd: list[str], cwd: str) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl training dataset path")
    ap.add_argument("--run_id", default=None, help="experiment id; outputs are isolated under artifacts/experiments/<run_id>/")
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--config_file", default="configs/model_dev.yaml")
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--resume", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--skip_quality_gate", action="store_true")
    ap.add_argument("--required_fields", default="message,system")
    args = ap.parse_args()

    root = _repo_root()
    _ensure_import_path(root)
    from sllm.common.experiments import build_run_paths, sanitize_run_id

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(os.path.join(root, input_path))

    run_id = args.run_id
    if run_id is not None:
        run_id = sanitize_run_id(run_id)
    if run_id:
        run_paths = build_run_paths(run_id, root_dir=os.path.join(root, "artifacts", "experiments"))
        tokenizer_dir = args.tokenizer_dir or os.path.relpath(run_paths.tokenizer_dir, root)
        model_dir = args.model_dir or os.path.relpath(run_paths.model_dir, root)
    else:
        tokenizer_dir = args.tokenizer_dir or "artifacts/tokenizer"
        model_dir = args.model_dir or "artifacts/model_dev"

    model_dir_abs = os.path.abspath(os.path.join(root, model_dir))
    os.makedirs(model_dir_abs, exist_ok=True)

    if not args.skip_quality_gate:
        quality_report = os.path.join(model_dir_abs, "data_quality_report.json")
        _run(
            [
                sys.executable,
                "-m",
                "sllm.train.data_quality_gate",
                "--input_source",
                input_path,
                "--output_file",
                quality_report,
                "--required_fields",
                args.required_fields,
                "--fail_on_fail",
            ],
            cwd=root,
        )

    run_meta_path = os.path.join(model_dir_abs, "run_meta.json")
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id or "__default__",
                "input_source": input_path,
                "tokenizer_dir": tokenizer_dir,
                "model_dir": model_dir,
                "train_ratio": args.train_ratio,
                "resume": args.resume,
                "device": args.device,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[meta] saved:", run_meta_path, flush=True)

    # Train tokenizer if missing.
    tok_json = os.path.join(root, tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
        _run(
            [
                sys.executable,
                "-m",
                "sllm.tokenizer.train_tokenizer",
                "--train_file",
                input_path,
                "--auto_valid_ratio",
                "0.2",
                "--output_dir",
                tokenizer_dir,
                "--vocab_size",
                str(args.vocab_size),
                "--resume",
                args.resume,
            ],
            cwd=root,
        )
    else:
        print("[skip] tokenizer exists:", tok_json, flush=True)

    # Train model with 80/20 auto split (same stable split as evaluate).
    _run(
        [
            sys.executable,
            "-m",
            "sllm.train.train_decoder",
            "--train_file",
            input_path,
            "--tokenizer_dir",
            tokenizer_dir,
            "--config_file",
            args.config_file,
            "--output_dir",
            model_dir,
            "--device",
            args.device,
            "--resume",
            args.resume,
            "--auto_split",
            "--train_ratio",
            str(args.train_ratio),
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "sllm.train.evaluate",
            "--train_file",
            input_path,
            "--auto_split",
            "--train_ratio",
            str(args.train_ratio),
            "--tokenizer_dir",
            tokenizer_dir,
            "--model_dir",
            model_dir,
            "--device",
            args.device,
            "--resume",
            args.resume,
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "sllm.train.visualize_data_distribution",
            "--train_file",
            input_path,
            "--auto_split",
            "--train_ratio",
            str(args.train_ratio),
            "--output_dir",
            model_dir,
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "sllm.train.check_reason_coverage",
            "--model_dir",
            model_dir,
            "--input_file",
            input_path,
        ],
        cwd=root,
    )

    # Refresh summary report.
    _run([sys.executable, "-m", "sllm.train.report_results", "--model_dir", model_dir], cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
