import argparse
import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, StripAccents, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from sllm.common.io import (
    count_jsonl_lines_in_source,
    extract_input_fields,
    flatten_input_fields,
    iter_jsonl_source,
    read_jsonl,
    resolve_jsonl_source_paths,
)
from sllm.common.prompting import build_prompt, build_target
from sllm.common.modeling import SPECIAL_TOKENS


def _split_sizes(total: int, valid_ratio: float):
    """Return (n_train_rows, n_valid_rows) for approximately valid_ratio valid."""
    if total <= 0:
        return 0, 0
    n_valid = int(round(total * float(valid_ratio)))
    n_valid = max(0, min(n_valid, total))
    if n_valid >= total:
        n_valid = max(0, total - 1)
    n_train = total - n_valid
    return n_train, n_valid


def _iter_rows_train_then_valid_source(source: str, n_train: int):
    """First n_train rows, then the rest (merged file order). Two passes, no temp files."""
    for i, row in enumerate(iter_jsonl_source(source)):
        if i >= n_train:
            break
        yield row
    for i, row in enumerate(iter_jsonl_source(source)):
        if i < n_train:
            continue
        yield row


def iter_texts_from_rows(rows_iter, start_row=0, progress_path=None, progress_meta=None, save_every=2000):
    row_index = 0
    for row in rows_iter:
        if row_index < start_row:
            row_index += 1
            continue
        yield flatten_input_fields(extract_input_fields(row))
        yield build_prompt(row)
        if isinstance(row, dict) and ("command" in row and "reason" in row):
            yield build_target(row)
            if row.get("command") is not None:
                yield str(row.get("command"))
            if row.get("reason") is not None:
                yield str(row.get("reason"))
        row_index += 1
        if progress_path and row_index % save_every == 0:
            state = {"consumed_rows": row_index, "completed": False}
            if progress_meta:
                state.update(progress_meta)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)


def iter_texts(files, start_row=0, progress_path=None, progress_meta=None, save_every=2000):
    def all_rows():
        for path in files:
            yield from read_jsonl(path)

    yield from iter_texts_from_rows(
        all_rows(),
        start_row=start_row,
        progress_path=progress_path,
        progress_meta=progress_meta,
        save_every=save_every,
    )


def init_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.normalizer = Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def count_rows(files):
    total = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_file",
        required=True,
        help="JSONL file or directory (recursive *.jsonl, sorted path merge order)",
    )
    ap.add_argument(
        "--valid_file",
        default=None,
        help="optional extra JSONL for vocabulary; if omitted, only --train_file is used",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    ap.add_argument(
        "--auto_valid_ratio",
        type=float,
        default=None,
        help=(
            "if set, use this fraction of --train_file rows as tokenizer valid corpus "
            "(file order: train segment first, then valid). Mutually exclusive with --valid_file."
        ),
    )
    args = ap.parse_args()

    if args.auto_valid_ratio is not None and args.valid_file:
        raise ValueError("use either --valid_file or --auto_valid_ratio, not both")
    if args.auto_valid_ratio is not None and not (0.0 < args.auto_valid_ratio < 1.0):
        raise ValueError("--auto_valid_ratio must be strictly between 0 and 1")

    if args.valid_file and not os.path.isfile(args.valid_file):
        raise FileNotFoundError(
            f"valid_file not found: {args.valid_file!r} (omit --valid_file to use train_file only)"
        )

    n_train = None
    if args.auto_valid_ratio is not None:
        total_rows = count_jsonl_lines_in_source(args.train_file)
        n_train, _n_valid = _split_sizes(total_rows, args.auto_valid_ratio)
        files = []
    else:
        files = resolve_jsonl_source_paths(args.train_file) + (
            [args.valid_file] if args.valid_file else []
        )
        total_rows = count_rows(files)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    progress_path = os.path.join(args.output_dir, "tokenizer_resume_state.json")

    progress = None
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)

    start_row = 0
    if args.resume != "never":
        if progress:
            start_row = int(progress.get("consumed_rows", 0))
            if progress.get("completed") and os.path.exists(tokenizer_path):
                print("tokenizer already completed:", tokenizer_path)
                print("saved:", tokenizer_path)
                return
        elif os.path.exists(tokenizer_path):
            if args.resume == "always":
                raise RuntimeError("resume requested but tokenizer_resume_state.json not found")
            print("tokenizer exists; skip retraining in auto resume mode:", tokenizer_path)
            print("saved:", tokenizer_path)
            return

    if args.resume == "never":
        tokenizer = init_tokenizer()
        start_row = 0
    else:
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.normalizer = Sequence([NFD(), StripAccents()])
            tokenizer.pre_tokenizer = ByteLevel()
            tokenizer.decoder = ByteLevelDecoder()
        else:
            if args.resume == "always":
                raise RuntimeError("resume requested but tokenizer.json not found")
            tokenizer = init_tokenizer()

    if start_row >= total_rows:
        tokenizer.save(tokenizer_path)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_file": args.train_file,
                    "valid_file": args.valid_file,
                    "auto_valid_ratio": args.auto_valid_ratio,
                    "tokenizer_split_train_rows": n_train,
                    "vocab_size": args.vocab_size,
                    "total_rows": total_rows,
                    "consumed_rows": total_rows,
                    "completed": True,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("tokenizer already up-to-date:", tokenizer_path)
        print("saved:", tokenizer_path)
        return

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[
            SPECIAL_TOKENS["pad_token"],
            SPECIAL_TOKENS["bos_token"],
            SPECIAL_TOKENS["eos_token"],
            SPECIAL_TOKENS["unk_token"],
        ],
        initial_alphabet=ByteLevel.alphabet(),
    )
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_file": args.train_file,
                "valid_file": args.valid_file,
                "auto_valid_ratio": args.auto_valid_ratio,
                "tokenizer_split_train_rows": n_train,
                "vocab_size": args.vocab_size,
                "total_rows": total_rows,
                "consumed_rows": start_row,
                "completed": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    progress_meta = {
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "vocab_size": args.vocab_size,
        "total_rows": total_rows,
    }
    if args.auto_valid_ratio is not None:
        progress_meta["auto_valid_ratio"] = args.auto_valid_ratio
        progress_meta["tokenizer_split_train_rows"] = n_train
        progress_meta["tokenizer_split_valid_rows"] = total_rows - n_train

    if args.auto_valid_ratio is not None:
        print(
            f"tokenizer corpus split from {args.train_file}: "
            f"{n_train} train rows + {total_rows - n_train} valid rows "
            f"(auto_valid_ratio={args.auto_valid_ratio})"
        )

    def corpus_iterator(srow):
        if args.auto_valid_ratio is not None:
            return iter_texts_from_rows(
                _iter_rows_train_then_valid_source(args.train_file, n_train),
                start_row=srow,
                progress_path=progress_path,
                progress_meta=progress_meta,
            )
        return iter_texts(
            files,
            start_row=srow,
            progress_path=progress_path,
            progress_meta=progress_meta,
        )

    tokenizer.train_from_iterator(corpus_iterator(start_row), trainer=trainer)
    tokenizer.save(tokenizer_path)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_file": args.train_file,
                "valid_file": args.valid_file,
                "auto_valid_ratio": args.auto_valid_ratio,
                "tokenizer_split_train_rows": n_train,
                "vocab_size": args.vocab_size,
                "total_rows": total_rows,
                "consumed_rows": total_rows,
                "completed": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("saved:", tokenizer_path)

if __name__ == "__main__":
    main()
