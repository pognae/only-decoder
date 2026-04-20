# AGENTS.md

## MUST READ FIRST
This repository is governed by this file.
When working in this repo, follow this document as the highest-priority project instruction file for local repository decisions.

If any code, README, comments, or prior assumptions conflict with this file, follow **this file**.

---

## 1. PROJECT IDENTITY

This project is a **single-model Action SLLM** for legacy systems.

It is **not**:
- a chatbot
- a generic assistant
- a two-model classifier + generator system
- a RAG system (at the current phase)
- a multi-agent framework (at the current phase)
- a workflow orchestration platform
- a general-purpose LLM product

It **is**:
- a **single decoder-only Transformer** system
- trained to read full legacy-system input
- and return a **single fixed JSON output**
- containing:
  - `command`
  - `reason`
  - `accuracy`

The primary business goal is:
- connect many legacy systems
- infer what action is needed
- generate the correct command
- explain the reason briefly in natural language
- expose the model through an external API

---

## 2. CURRENT PHASE VS FUTURE PHASE

### 2.1 Current phase (locked)
In the current phase, the repository must focus on:

```text
legacy input
→ single decoder-only model
→ JSON output
```

During the current phase:
- **RAG is not part of the implementation**
- **multi-agent architecture is not part of the implementation**
- **only one final model is allowed**

### 2.2 Future phase (allowed later)
After the current single-model source is considered complete and stable, the project may later be expanded with:
- RAG
- multi-agent orchestration

But those are **future extensions only**.

### 2.3 Rule
Do **not** add RAG or multi-agent architecture during the current implementation phase unless the user explicitly says:
- the current source is complete
- and it is now time to add the extension

---

## 3. FINAL ARCHITECTURE DECISIONS (LOCKED)

These decisions are already made and must be treated as fixed unless the user explicitly changes them.

### 3.1 Language
- Use **Python**
- Prefer Python libraries aggressively
- Avoid unnecessary low-level implementation when reliable libraries exist

### 3.2 Model architecture
- Use **ONE final model only**
- Use **decoder-only Transformer**
- Do **not** split into BERT + decoder
- Do **not** introduce an extra classifier model
- Do **not** introduce a separate generator model

### 3.3 Output format
Output must always be a **single JSON object**.

Required fields:
- `command`
- `reason`
- `accuracy`

### 3.4 Natural language
- `reason` must contain a human-readable natural-language explanation
- This explanation should be short, useful, and domain-relevant
- `reason` must be intent-grounded from the current input message/fields
- avoid returning fixed catalog template phrases verbatim when input context is available
- `reason` text must remain semantically aligned with the selected `command`
- Do not output free-form paragraphs outside JSON

### 3.5 Data assumptions
- There are about **5,000,000 records**
- All records are labeled
- Records may vary in type and schema
- Legacy input may be structured, semi-structured, or string-based
- Input diversity is expected

### 3.6 Infrastructure assumptions
- GPU count available: **1**
- Development speed is the highest priority
- This repo will be worked on continuously through Codex
- The model must expose a public-facing API endpoint for legacy systems

### 3.7 Training objective
The model must learn:

```text
legacy data in
→ command + reason out
```

### 3.8 Forbidden architecture changes
Do **not** change this project into:
- BERT + decoder
- encoder-decoder T5 style
- RAG during the current phase
- vector search first architecture during the current phase
- LangChain-first orchestration during the current phase
- multi-model routing
- tool-agent decision graph during the current phase
- classifier first / generator second system

Unless the user explicitly instructs otherwise, all implementation work must preserve the **single-model decoder-only** design.

---

## 4. DOMAIN COVERAGE (EXPANDED)

This project is **not limited to logistics**.

The legacy systems connected to this SLLM may belong to any of these domains:

- logistics
- renewable energy
- bio / biotech / biomanufacturing
- transportation
- ERP
- MES

### 4.1 Domain principle
Treat this as a **multi-domain legacy action model**.

This means:
- prompts
- training format
- inference logic
- schema design
- reason generation
- command generation

must all support cross-domain input.

### 4.2 Domain-safe implementation rule
Never hardcode the project as logistics-only unless the user explicitly asks for a logistics-only specialization.

### 4.3 Recommended input framing
Each record should support at least:
- source domain
- source system
- raw message or structured legacy fields
- optional metadata
- optional identifiers

Example conceptual input:

```json
{
  "domain": "mes",
  "system": "MES01",
  "message": "temperature deviation detected",
  "batch_id": "B-1120"
}
```

---

## 5. PRODUCT CONTRACT

### 5.1 Required inference contract
Every successful inference response must return JSON in this shape:

```json
{
  "command": "UPDATE_LOCATION",
  "reason": "location master data missing",
  "accuracy": {
    "command_accuracy": 0.84
  }
}
```

### 5.2 Response rules
- Return one JSON object only
- No markdown in API responses
- No explanatory wrapper text in API responses
- No extra keys unless the user explicitly requests them
- Keep field names stable
- Keep JSON machine-readable and deterministic

### 5.3 Failure-safe response
If generation or parsing fails, degrade safely:

```json
{
  "command": "NO_ACTION",
  "reason": "model output could not be parsed safely",
  "accuracy": {
    "command_accuracy": null
  }
}
```

---

## 6. DEVELOPMENT PRIORITY ORDER (LOCKED)

When choosing between implementation options, use this priority order:

1. **development speed**
2. correctness of JSON output
3. simplicity
4. maintainability
5. reproducibility
6. training efficiency
7. inference speed
8. model size ambition

This means:
- prefer simple working code over elegant complex code
- prefer libraries over custom implementations
- prefer small stable iterations over ambitious redesigns
- prefer a reliable dev model over a broken large model

---

## 7. LIBRARY POLICY

Use Python libraries aggressively.

### 7.1 Preferred libraries
Use these first when appropriate:
- `transformers`
- `datasets`
- `tokenizers`
- `torch`
- `accelerate`
- `fastapi`
- `uvicorn`
- `pydantic`
- `pyyaml`

### 7.2 Do not reinvent unless necessary
Do not hand-write:
- tokenizer internals
- trainer loops
- config serialization
- API validation
- model loading utilities

unless there is a clear reason and the user specifically benefits.

### 7.3 Direct implementation boundary
The system may still be "ours" while using Python libraries.
Using `transformers` is acceptable.
Using a decoder-only architecture initialized and trained in our project is acceptable.
Do not interpret “our own SLLM” to mean “no libraries allowed.”

---

## 8. MODEL STRATEGY

### 8.1 Main model choice
Use a **decoder-only Transformer**, library-backed.

Preferred direction:
- `LlamaConfig` + `LlamaForCausalLM` from config
- initialized from scratch for our project flow
- trained on our labeled legacy data

### 8.2 Why this is required
Because one model must do both:
- command generation
- natural-language explanation

A decoder-only model matches this requirement best.

### 8.3 Model scaling rule
Use a **small dev configuration first**.
Only scale upward after:
- tokenizer works
- training pipeline works
- inference works
- API works
- evaluation works

### 8.4 1B rule
This project may target a larger model later, but development must start from a smaller dev config.
Do not jump directly to a 1B-sized training run before the full pipeline is validated.

---

## 9. DATA RULES

### 9.1 Canonical training shape
Training rows should conceptually look like:

```json
{
  "record_id": "1",
  "domain": "logistics",
  "system": "WMS",
  "message": "picking location not found",
  "warehouse_id": "WH01",
  "command": "UPDATE_LOCATION",
  "reason": "location master data missing"
}
```

Notes:
- `record_id` is **optional** metadata (helpful for tracing, but not required).
- `command`/`reason` are **labels** for supervised training rows. Some JSONL inputs (e.g. bronze/unlabeled data) may omit them.
- For inference/API input payloads, `command` is **not required** and is ignored if provided.

### 9.2 Prompting rule
Training should use:
- prompt = serialized legacy input centered on `message`
- target = JSON output

### 9.3 Keep all fields
Do not silently drop legacy fields unless there is a strong documented reason.
Legacy inputs may contain important predictive signals.

### 9.4 Stable serialization
Use a consistent serialization order for structured input.
If dicts are serialized, keep field order deterministic.

### 9.5 Mixed schema handling
Because source records come from multiple domains and systems:
- normalize as little as necessary
- preserve signal
- avoid domain-destructive flattening
- prefer consistent string serialization over aggressive schema rewriting

---

## 10. MULTI-DOMAIN LEGACY INPUT POLICY

This repo must support legacy input from:
- logistics
- renewable energy
- bio
- transportation
- ERP
- MES

### 10.1 Domain-aware prompting
When useful, include domain in the prompt.
Do not assume the domain can always be inferred from message text alone.

### 10.2 Domain expansion rule
Future code changes must not make the system less capable of supporting additional domains.

### 10.3 Do not overfit repo structure to one vertical
Avoid naming files or services in a logistics-only way if they represent core generic functionality.

Good:
- `message`
- `command`
- `reason`

Bad for core shared components:
- `wms_router`
- `logistics_only_parser`
- `warehouse_action_core`

unless the file is explicitly domain-specific and intentionally isolated.

---

## 11. REPOSITORY STRUCTURE

Expected structure:

```text
configs/
sample_data/
src/sllm/common/
src/sllm/tokenizer/
src/sllm/train/
src/sllm/infer/
src/sllm/api/
```

### 11.1 Directory responsibilities
- `configs/`: model/training YAML configs
- `sample_data/`: example JSONL training/validation data
- `src/sllm/common/`: shared utilities, prompting, IO, config helpers
- `src/sllm/tokenizer/`: tokenizer build/training code
- `src/sllm/train/`: training and evaluation code
- `src/sllm/infer/`: runtime inference wrapper
- `src/sllm/api/`: FastAPI service layer

### 11.2 Do not sprawl the codebase
Avoid exploding this repo into many services.
This repo is primarily:
- one model
- one training pipeline
- one inference wrapper
- one API

---

## 12. CODING RULES

### 12.1 General style
- Prefer small files
- Prefer direct code over framework gymnastics
- Prefer readability over cleverness
- Avoid unnecessary abstraction layers
- Keep comments short and useful
- Avoid “magic” behavior

### 12.2 Configuration
- Use YAML or environment variables
- Avoid hardcoding environment-specific values
- Keep dev config and larger config separate

### 12.3 Determinism
Default inference should be deterministic.
If sampling is added, it must be opt-in and clearly justified.

### 12.4 JSON safety
Any code path that affects model output parsing must preserve:
- stable JSON extraction
- safe fallback
- no markdown wrappers

---

## 13. TRAINING RULES

### 13.1 Expected pipeline
1. tokenizer training
2. small dev model training
3. evaluation
4. inference smoke test
5. API smoke test
6. larger config only after pipeline is stable

### 13.2 Do not skip the dev model phase
Never begin by training the biggest configuration first.

### 13.3 If memory is tight
Reduce in this order:
- batch size
- sequence length
- hidden size
- number of layers

Do not redesign the architecture prematurely.

### 13.4 Training objective
Use supervised generation with:
- prompt = legacy input centered on `message`
- target = JSON

Practical rule:
- The primary predictive signal is typically `message`.
- Training must learn to infer the output `command` from input `message`.
- Non-message input fields are primarily preserved to help produce a richer and more faithful `reason`.
- Do **not** rely on `record_id` or `command` being present in the input payload; `record_id` is optional metadata and `command` is a label (may be absent in unlabeled JSONL inputs).

### 13.5 Accuracy handling
Accuracy values are evaluation metrics from held-out validation, not per-example certainty.
Keep that distinction clear in code and docs.

---

## 14. API RULES

### 14.1 Required API purpose
External legacy systems will call this model over HTTP.

### 14.2 Required endpoint set
Keep the API minimal:
- `/health`
- `/infer`
- `/commands` (browser command catalog management page)

Add more endpoints only if clearly necessary.

### 14.3 `/infer` rules
Input:
- accept top-level input record fields (string or dict)
- allow string or dict input
- validate structure through Pydantic
- do **not** require `command` in input
- ignore/drop `command`, `reason`, `accuracy` if sent in input payload

Output:
- return JSON matching the product contract
- no markdown
- no prose wrapper
- no debugging text
- include `command` only in output (model prediction + catalog normalization)
- include `reason` in output only
- include `accuracy` in output only
- reason generation should prefer input-intent-aware wording over static template wording

### 14.4 Backward compatibility
If the response schema changes, update:
- README
- API examples
- evaluation if needed
- parsing fallback logic

---

## 15. DOCUMENTATION RULES

Whenever changing any of the following:
- prompt format
- output schema
- tokenizer behavior
- model config
- preprocessing
- API format
- evaluation metrics

also update:
- `README.md`
- sample data if needed
- config comments if needed

Do not let docs drift from code.

---

## 16. VALIDATION CHECKLIST

After meaningful changes, run the smallest relevant end-to-end checks.

### Minimum expected checks
1. tokenizer training command
2. model training command on sample data
3. evaluation command
4. API startup
5. one `/infer` request

If a full training run is too expensive, run a smoke test and state that clearly.

---

## 17. COMMANDS

### Install
```bash
pip install -r requirements.txt
```

### Set import path
#### Windows
```bash
set PYTHONPATH=src
```

#### Linux/Mac
```bash
export PYTHONPATH=src
```

### Train tokenizer
`make tok` uses `--auto_valid_ratio` (default **0.2** via `TOK_VALID_RATIO`) to take that fraction of `TRAIN_FILE` rows as tokenizer valid corpus (file order: train segment first, then valid). No separate `valid.jsonl` is required for tokenizer training.

```bash
python -m sllm.tokenizer.train_tokenizer --train_file sample_data/train.jsonl --auto_valid_ratio 0.2 --output_dir artifacts/tokenizer --vocab_size 4000
```

Optional explicit valid JSONL (mutually exclusive with `--auto_valid_ratio`):

```bash
python -m sllm.tokenizer.train_tokenizer --train_file sample_data/train.jsonl --valid_file sample_data/valid.jsonl --output_dir artifacts/tokenizer --vocab_size 4000
```

### Train model
Default is **80% train / 20% validation** from a single JSONL (or **a directory**: all `*.jsonl` recursively, merged in sorted path order) via stable hash split (`--auto_split --train_ratio 0.8`). `make train` runs **tokenizer → train_decoder → evaluate** in one flow; `RESUME=auto` continues from checkpoints when present.

```bash
make train TRAIN_FILE=sample_data/train.jsonl
# or: every sample_data/**/*.jsonl merged
make train TRAIN_FILE=sample_data
```

Equivalent manual commands:

```bash
python -m sllm.train.train_decoder --train_file sample_data/train.jsonl --tokenizer_dir artifacts/tokenizer --config_file configs/model_dev.yaml --output_dir artifacts/model_dev --auto_split --train_ratio 0.8
```

Optional separate validation file (no auto split):

```bash
python -m sllm.train.train_decoder --train_file sample_data/train.jsonl --valid_file sample_data/valid.jsonl --tokenizer_dir artifacts/tokenizer --config_file configs/model_dev.yaml --output_dir artifacts/model_dev
```

### Evaluate
On the **same 20% held-out split** as training (when using `--auto_split`):

```bash
python -m sllm.train.evaluate --train_file sample_data/train.jsonl --auto_split --train_ratio 0.8 --tokenizer_dir artifacts/tokenizer --model_dir artifacts/model_dev
```

Or with an explicit validation JSONL:

```bash
python -m sllm.train.evaluate --valid_file sample_data/valid.jsonl --tokenizer_dir artifacts/tokenizer --model_dir artifacts/model_dev
```

### Serve API
```bash
uvicorn sllm.api.server:app --host 0.0.0.0 --port 8000
```

---

## 18. FUTURE EXTENSION POLICY

### 18.1 RAG
RAG is a **future extension**.
It is not part of the current implementation.
Add it only after:
- the single-model source is complete
- training/evaluation/API are stable
- the user explicitly asks to add RAG

### 18.2 Multi-agent
Multi-agent is also a **future extension**.
It is not part of the current implementation.
Add it only after:
- the single-model source is complete
- the user explicitly asks for the extension

### 18.3 Current rule
At the current phase:
- do not create RAG folders
- do not create vector DB code
- do not create agent orchestration layers
- do not redesign inference around multiple subcomponents

---

## 19. NON-GOALS

Unless explicitly requested by the user, do not turn this repo into:
- a logistics-only model
- a chatbot
- BERT + decoder
- RAG + generator during the current phase
- a multi-model platform
- a distributed serving stack
- a multi-agent control framework during the current phase
- a research playground for many architectures

This is a **production-oriented, development-speed-first, single-model legacy action SLLM**.

---

## 20. DECISION RULE WHEN UNSURE

If uncertain between multiple implementation choices, pick the option that best satisfies:

```text
single model
+ JSON stability
+ fast development
+ multi-domain legacy compatibility
+ simple Python library-based implementation
```

If still uncertain, do not broaden the architecture.
Stay within the current project contract.

---

## 21. FINAL PROJECT SUMMARY

This repo builds a **strong single-model Action SLLM** that:
- uses Python
- uses one decoder-only Transformer
- supports many legacy domains
- accepts external legacy input by API
- outputs command + natural-language reason in JSON
- uses labeled training data at large scale
- prioritizes development speed and practical execution
- may later be extended with RAG and multi-agent only after the current source is complete

This is the core project identity.
Do not weaken it.

---

## 22. EXPLICIT MESSAGE-CENTERED INPUT AND COMMAND-CATALOG RULES (ADDED)

This section is additive and high priority.
Keep all prior sections, but interpret them through the rules below when implementing data handling, training, inference, command selection, and documentation.

### 22.1 Message is the canonical primary input
The canonical primary input field is **`message`**.

Rules:
- input schema must be centered on `message`
- do **not** require `legacy_input.message`
- do **not** assume nested `legacy_input` structure
- do **not** make `legacy_input` the default contract
- top-level `message` is the default and preferred shape

Good:
```json
{
  "domain": "logistics",
  "system": "WMS01",
  "message": "scanner timeout detected on line 3",
  "site": "ICN-01"
}
```

Bad as the default contract:
```json
{
  "legacy_input": {
    "message": "scanner timeout detected on line 3"
  }
}
```

### 22.2 Message drives learning and command generation
The model must primarily learn from **`message`** and use the learned signal from `message` to generate:
- `command`
- `reason`

Implementation meaning:
- `message` is the primary predictive signal for `command`
- `message` is also the primary semantic anchor for `reason`
- other columns must not replace `message` as the main source of action selection

### 22.3 Non-message columns are for reason support, not primary command selection
All other input columns are secondary support fields.

Rules:
- non-message fields are mainly used to enrich and ground `reason`
- they may help preserve context fidelity
- but they must not replace `message` as the central action trigger
- command selection logic must remain message-centered by default

### 22.4 Reason must be meaningfully detailed
The output center remains **`command`**, but `reason` must explain why that command was selected.

Rules:
- `reason` should not be too short
- target a meaningful explanation of roughly **100+ characters** when the input provides enough context
- this is a quality target, not a hard parser constraint
- `reason` may be in **English by default**
- Korean is also allowed when needed or requested
- `reason` must remain semantically aligned with `command`

### 22.5 Command catalog must be checked and can change
Commands are user-managed and may change over time.

Therefore:
- implementation must support checking the latest **command catalog**
- model-facing command selection must align to the current catalog semantics
- both training-time and inference-time workflows may use the command catalog
- the system must not assume the command set is permanently fixed in code

### 22.6 Command selection must reflect catalog meaning
When choosing a command:
- first inspect the available command catalog
- understand command meaning from catalog metadata maintained by the user
- choose the command whose meaning best matches the message intent
- avoid stale hardcoded command assumptions when a current catalog is available

### 22.7 Command naming style
Commands are code-like identifiers, but they must remain readable and semantically meaningful to humans.

Preferred style:
- uppercase
- underscore-separated
- domain-revealing when helpful

Example:
- `LOGISTICS_RESET_SCANNER_CONNECTION`
- `BIO_HOLD_BATCH_PROCESS`
- `MES_UPDATE_LOT_STATUS`

Avoid opaque forms as the default public identifier:
- `CMD_00192`
- `A17X9`
- `DO_ACTION_7`

Opaque internal IDs may still exist, but the primary command field should be readable.

### 22.8 Accuracy remains required
The `accuracy` field remains part of the output contract.
Do not remove it.

### 22.9 Strong implementation rule
If any prior example, helper, sample payload, prompt, or preprocessing logic suggests a nested `legacy_input.message`-first contract, replace that assumption with a **top-level `message`-first contract** unless the user explicitly asks otherwise.
