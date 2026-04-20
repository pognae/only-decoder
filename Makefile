PYTHON ?= python3
ifneq ($(wildcard .venv/bin/python),)
PYTHON = .venv/bin/python
endif
PY := PYTHONPATH=src $(PYTHON)

# -----------------------------
# Performance-oriented env vars
# -----------------------------
# These do not change model architecture; they tune runtime behavior.
# Override per-run: e.g. `make train OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
export HF_HUB_DISABLE_TELEMETRY ?= 1
export TRANSFORMERS_NO_ADVISORY_WARNINGS ?= 1
export TOKENIZERS_PARALLELISM ?= false
# Helps reduce CUDA memory fragmentation on long runs (safe no-op on CPU/MPS).
export PYTORCH_CUDA_ALLOC_CONF ?= expandable_segments:True
# Optional: set for CPU-heavy tokenization/preprocess (only exported if non-empty).
OMP_NUM_THREADS ?=
MKL_NUM_THREADS ?=
NUMEXPR_NUM_THREADS ?=
ifneq ($(strip $(OMP_NUM_THREADS)),)
export OMP_NUM_THREADS
endif
ifneq ($(strip $(MKL_NUM_THREADS)),)
export MKL_NUM_THREADS
endif
ifneq ($(strip $(NUMEXPR_NUM_THREADS)),)
export NUMEXPR_NUM_THREADS
endif

TRAIN_FILE ?= train_data/train.jsonl
VALID_FILE ?= train_data/valid.jsonl
RUN_ID ?=
ifeq ($(strip $(RUN_ID)),)
TOKENIZER_DIR ?= artifacts/tokenizer
MODEL_DIR ?= artifacts/model_dev
else
TOKENIZER_DIR ?= artifacts/experiments/$(RUN_ID)/tokenizer
MODEL_DIR ?= artifacts/experiments/$(RUN_ID)/model
endif
CONFIG_FILE ?= configs/model_dev.yaml
INFER_INPUT ?= sample_data/infer_input.json
BRONZE_FILE ?= train_data/bronze_mixed_3000.jsonl
SILVER_FILE ?= train_data/silver_mixed_3000.jsonl

VOCAB_SIZE ?= 4000
DEVICE ?= auto
# auto: continue tokenizer/model/eval from checkpoints when present; never: always restart
RESUME ?= auto
MIN_CONF ?= 0.8
BATCH_SIZE ?= 16
HOST ?= 0.0.0.0
PORT ?= 8000
# 1: 80%% train / 20%% valid from TRAIN_FILE (stable hash, same split for train/eval/infer reference)
AUTO_SPLIT ?= 1
TRAIN_RATIO ?= 0.8
TOK_VALID_RATIO ?= 0.2
REQUIRED_FIELDS ?= message,system
FINALIZE_RUN_ID ?=
MIN_COMMAND_ACCURACY ?=
MIN_REASON_COVERAGE ?= 1.0
MAX_COMMAND_MALFORMED_RATIO ?= 0.0
DRY_RUN ?= 0
REQUIRE_REVIEW_APPROVAL ?= 1

# Flags for evaluate / infer / report / coverage when using automatic split
EVAL_SPLIT_FLAGS = $(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),--train_file $(TRAIN_FILE) --auto_split --train_ratio $(TRAIN_RATIO),--valid_file $(VALID_FILE))
DIST_FLAGS = $(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),--auto_split --train_ratio $(TRAIN_RATIO),$(if $(wildcard $(VALID_FILE)),--valid_file $(VALID_FILE),))
COVERAGE_INPUT = $(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),$(TRAIN_FILE),$(VALID_FILE))

.PHONY: help quality tok train eval report infer api b2s coverage dist filter-reason finalize reset-train reset-train-all

help:
	@echo "Short commands:"
	@echo "  make tok            # tokenizer on TRAIN_FILE (TOK_VALID_RATIO for vocab valid slice)"
	@echo "  make train          # tok + model train + eval (80/20 from TRAIN_FILE when AUTO_SPLIT=1)"
	@echo "  make eval           # evaluate on the same valid split as training"
	@echo "  make report         # 1-input report_result"
	@echo "  make infer          # infer JSON input"
	@echo "  make api            # run FastAPI server"
	@echo "  make b2s            # bronze -> silver"
	@echo "  make coverage       # reason coverage check"
	@echo "  make dist           # data distribution report"
	@echo "  make filter-reason  # drop rows with missing reason"
	@echo "  make finalize       # quality 통과 RUN을 기본 모델 경로로 최종 반영"
	@echo "  make reset-train    # delete current MODEL_DIR and TOKENIZER_DIR"
	@echo "  make reset-train-all# reset-train + delete artifacts/experiments/*"
	@echo ""
	@echo "Defaults: AUTO_SPLIT=1 TRAIN_RATIO=0.8 RESUME=auto"
	@echo "RUN_ID set example: make train RUN_ID=run_20260414_103000"
	@echo "Override example:"
	@echo "  make train DEVICE=gpu RESUME=never MODEL_DIR=artifacts/model_new"
	@echo "  make train AUTO_SPLIT=0 VALID_FILE=path/to/valid.jsonl"

quality:
	$(PY) -m sllm.train.data_quality_gate \
		--input_source $(TRAIN_FILE) \
		--output_file $(MODEL_DIR)/data_quality_report.json \
		--required_fields $(REQUIRED_FIELDS)

tok:
	$(PY) -m sllm.tokenizer.train_tokenizer \
		--train_file $(TRAIN_FILE) \
		--auto_valid_ratio $(TOK_VALID_RATIO) \
		--output_dir $(TOKENIZER_DIR) \
		--vocab_size $(VOCAB_SIZE) \
		--resume $(RESUME)

train: quality tok
	$(PY) -m sllm.train.train_decoder \
		--train_file $(TRAIN_FILE) \
		--tokenizer_dir $(TOKENIZER_DIR) \
		--config_file $(CONFIG_FILE) \
		--output_dir $(MODEL_DIR) \
		--device $(DEVICE) \
		--resume $(RESUME) \
		$(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),--auto_split,) \
		--train_ratio $(TRAIN_RATIO) \
		$(if $(filter 0 false FALSE no NO,$(AUTO_SPLIT)),--valid_file $(VALID_FILE),)
	$(PY) -m sllm.train.evaluate \
		--tokenizer_dir $(TOKENIZER_DIR) \
		--model_dir $(MODEL_DIR) \
		--device $(DEVICE) \
		--resume $(RESUME) \
		--batch_size $(BATCH_SIZE) \
		$(EVAL_SPLIT_FLAGS)

eval:
	$(PY) -m sllm.train.evaluate \
		--tokenizer_dir $(TOKENIZER_DIR) \
		--model_dir $(MODEL_DIR) \
		--device $(DEVICE) \
		--resume $(RESUME) \
		--batch_size $(BATCH_SIZE) \
		$(EVAL_SPLIT_FLAGS)

report:
	$(PY) -m sllm.train.report_result \
		--model_dir $(MODEL_DIR) \
		--input_file $(INFER_INPUT) \
		$(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),--train_file $(TRAIN_FILE) --auto_split --train_ratio $(TRAIN_RATIO),--valid_file $(VALID_FILE))

infer:
	$(PY) -m sllm.infer.run_infer_json \
		--model_dir $(MODEL_DIR) \
		--input_file $(INFER_INPUT) \
		$(if $(filter 1 true TRUE yes YES,$(AUTO_SPLIT)),--train_file $(TRAIN_FILE) --auto_split --train_ratio $(TRAIN_RATIO),--valid_file $(VALID_FILE))

api:
	SLLM_MODEL_DIR=$(MODEL_DIR) SLLM_TOKENIZER_DIR=$(MODEL_DIR) \
	$(PY) -m uvicorn sllm.api.server:app --host $(HOST) --port $(PORT)

b2s:
	$(PY) -m sllm.train.bronze_to_silver \
		--bronze_file $(BRONZE_FILE) \
		--silver_file $(SILVER_FILE) \
		--append \
		--min_confidence $(MIN_CONF)

coverage:
	$(PY) -m sllm.train.check_reason_coverage \
		--model_dir $(MODEL_DIR) \
		--input_file $(COVERAGE_INPUT)

dist:
	$(PY) -m sllm.train.visualize_data_distribution \
		--train_file $(TRAIN_FILE) \
		$(DIST_FLAGS) \
		--output_dir $(MODEL_DIR)

filter-reason:
	$(PY) -m sllm.train.filter_missing_reason \
		--input_file $(TRAIN_FILE) \
		--output_file train_data/train_no_missing_reason.jsonl

finalize:
	@if [ -z "$(strip $(FINALIZE_RUN_ID))" ]; then echo "FINALIZE_RUN_ID is required (example: make finalize FINALIZE_RUN_ID=run_20260414_103000)"; exit 2; fi
	$(PY) -m sllm.train.finalize_run \
		--run_id $(FINALIZE_RUN_ID) \
		--target_model_dir $(MODEL_DIR) \
		--target_tokenizer_dir $(TOKENIZER_DIR) \
		$(if $(strip $(MIN_COMMAND_ACCURACY)),--min_command_accuracy $(MIN_COMMAND_ACCURACY),) \
		--min_reason_coverage $(MIN_REASON_COVERAGE) \
		--max_command_malformed_ratio $(MAX_COMMAND_MALFORMED_RATIO) \
		$(if $(filter 0 false FALSE no NO,$(REQUIRE_REVIEW_APPROVAL)),--skip_review_approval,) \
		$(if $(filter 1 true TRUE yes YES,$(DRY_RUN)),--dry_run,)

reset-train:
	python3 scripts/reset_training_artifacts.py \
		--model_dir $(MODEL_DIR) \
		--tokenizer_dir $(TOKENIZER_DIR)

reset-train-all:
	python3 scripts/reset_training_artifacts.py \
		--model_dir $(MODEL_DIR) \
		--tokenizer_dir $(TOKENIZER_DIR) \
		--delete_experiments
