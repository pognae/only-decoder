1. 압축 해제
unzip legacy_action_sllm_single_model.zip
cd legacy_action_sllm_single_model

2. 패키지 설치
pip install -r requirements.txt

3. Python 경로 설정
- Windows
set PYTHONPATH=src

- Linux/Mac
export PYTHONPATH=src

짧은 명령어(권장):
- `make quality`
- `make tok`
- `make train` (내부에서 `make quality` → `make tok` → 모델 학습 → `evaluate`까지 한 번에 실행)
- `make eval` (학습과 동일한 20% 홀드아웃 기준, `AUTO_SPLIT=1`일 때)
- `make report`
- `make infer`
- `make api`
- `make b2s`
- `make coverage`
- `make dist`
- `make filter-reason`
- `make finalize`
- `make reset-train` (현재 `MODEL_DIR`, `TOKENIZER_DIR` 삭제 후 완전 초기화)
- `make reset-train-all` (위 + `artifacts/experiments/*`까지 삭제)

Make 변수(필요할 때만 override):
- `RUN_ID` (설정 시 `MODEL_DIR`=`artifacts/experiments/<RUN_ID>/model`, `TOKENIZER_DIR`=`artifacts/experiments/<RUN_ID>/tokenizer`)
- `MODEL_DIR` (기본: `artifacts/model_dev`)
- `TOKENIZER_DIR` (기본: `artifacts/tokenizer`)
- `TRAIN_FILE`: **단일 `.jsonl` 파일**이거나, **디렉터리**면 그 아래(하위 폴더 포함) 모든 `*.jsonl`을 **경로명 정렬 순**으로 이어 붙여 한 데이터셋처럼 사용합니다.
- `VALID_FILE` ( `AUTO_SPLIT=0` 일 때 등)
- `TOK_VALID_RATIO` (기본: `0.2`, `make tok`에서 `TRAIN_FILE`의 해당 비율을 토크나이저용 valid 코퍼스로 사용)
- `DEVICE` (`auto|cpu|gpu`)
- `RESUME` (`auto|always|never`)
- `AUTO_SPLIT` (`1`이면 **80/20** 자동 분할, `0`이면 `VALID_FILE` 사용)
- `TRAIN_RATIO` (기본: `0.8` = 학습 80% / 검증·평가 20%, `record_id` 등 기준 **안정 해시** 분할)
- `REQUIRED_FIELDS` (기본: `message,system`, 데이터 품질 게이트의 필수 컬럼)
- `FINALIZE_RUN_ID` (`make finalize` 대상 RUN_ID)
- `MIN_COMMAND_ACCURACY` (선택, 최종 반영 최소 정확도)
- `MIN_REASON_COVERAGE` (기본: `1.0`, 최종 반영 최소 reason coverage)
- `MAX_COMMAND_MALFORMED_RATIO` (기본: `0.0`, 최종 반영 허용 malformed ratio 상한)
- `DRY_RUN` (`1`이면 반영 없이 검증만 수행)
- `REQUIRE_REVIEW_APPROVAL` (기본: `1`, `make finalize` 시 검수 승인 필수)
- `HOST` / `PORT` (API 실행용)

학습 속도 튜닝(아키텍처 변경 없음, 환경변수만):
- 기본으로 아래 환경변수가 `make` 실행 시 export 됩니다(필요하면 override 가능).
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: CUDA 메모리 단편화 완화(장시간 학습 안정/속도에 도움)
  - `TOKENIZERS_PARALLELISM=false`: 토크나이저 병렬 스레드로 인한 과도한 경쟁/경고 방지(상황에 따라 true가 더 빠를 수 있음)
  - `HF_HUB_DISABLE_TELEMETRY=1`, `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`: 불필요한 부가 동작/출력 최소화
- CPU 쪽 전처리/토크나이즈가 병목이면 아래를 직접 설정해 실험해볼 수 있습니다.
  - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`
  - 예: `make train OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`

완전 초기화(학습 산출물 삭제):
- `make reset-train`: 현재 `MODEL_DIR`, `TOKENIZER_DIR`를 삭제해서 학습/토크나이저를 완전히 새로 시작합니다.
- `make reset-train-all`: 위 + `artifacts/experiments/*`까지 모두 삭제합니다.

4. 토크나이저 학습
- `make tok`: `TRAIN_FILE`만으로 학습하며, 기본 **20%**(`TOK_VALID_RATIO`)를 파일 순서상 뒷부분을 토크나이저용 valid 코퍼스로 씁니다.
- 수동 실행 예(동일 동작):

python3 -m sllm.tokenizer.train_tokenizer --train_file sample_data/train.jsonl --auto_valid_ratio 0.2 --output_dir artifacts/tokenizer --vocab_size 4000

- 별도 valid JSONL만 쓰려면 `--valid_file`을 주고 `--auto_valid_ratio`는 생략합니다.

4-1. 데이터 품질 게이트(자동화)
- 학습 전 품질 점검(필수 컬럼/중복/충돌/분포 편향 경고)을 JSON 리포트로 저장합니다.

python3 -m sllm.train.data_quality_gate --input_source sample_data/train.jsonl --output_file artifacts/model_dev/data_quality_report.json --required_fields message,system

- `status=PASS`여야 학습 진행을 권장합니다.
- `scripts/train.py`를 통한 학습(웹 업로드 포함)은 품질 게이트를 **자동 실행**하고, `FAIL`이면 학습을 중단합니다.

5. 모델 학습
- 기본(권장): `TRAIN_FILE` 하나로 **80% 학습 / 20% 검증**(안정 해시 분할). `make train`은 `quality gate` → 토크나이저 학습 → 모델 학습 → 평가까지 실행합니다.
  - `make train` 또는 `python3 -m sllm.train.train_decoder --train_file ... --auto_split --train_ratio 0.8 ...`
- 기존 방식(별도 valid 파일 사용):
  - `AUTO_SPLIT=0` 또는 `--auto_split` 없이 `--valid_file` 지정
- 소량 데이터(예: 1건 업로드)로 `--auto_split` 시 검증 split이 비면, 학습 파이프라인은 학습 샘플 1건을 검증 smoke set으로 자동 대체해 학습이 중단되지 않도록 동작합니다.
- `train_decoder`의 Hugging Face datasets 캐시는 기본적으로 `artifacts/_hf_datasets_cache`를 사용합니다. 필요 시 `SLLM_HF_DATASETS_CACHE`로 경로를 바꿀 수 있습니다.

예시(기본 자동 분할):
python3 -m sllm.train.train_decoder --train_file sample_data/train.jsonl --tokenizer_dir artifacts/tokenizer --config_file configs/model_dev.yaml --output_dir artifacts/model_dev --auto_split --train_ratio 0.8

예시(별도 valid 파일):
python3 -m sllm.train.train_decoder --train_file sample_data/train.jsonl --valid_file sample_data/valid.jsonl --tokenizer_dir artifacts/tokenizer --config_file configs/model_dev.yaml --output_dir artifacts/model_dev

실험 분리(RUN_ID) 예시:
- `scripts/train.py --run_id <RUN_ID>`를 쓰면 산출물이 `artifacts/experiments/<RUN_ID>/`로 분리됩니다.

.venv/bin/python scripts/train.py --input sample_data/train.jsonl --run_id run_20260414_103000

RUN_ID 저장/갱신 규칙(중요):
- `새 run_id`로 학습: 결과가 run_id별로 **누적 저장**됩니다. (`artifacts/experiments/<run_id>/`)
- `같은 run_id`로 재학습: 해당 run_id 결과가 **갱신/이어학습**됩니다. (`resume` 설정 영향)
- `run_id` 없이 학습: 기본 경로(`artifacts/model_dev`)가 **갱신**됩니다.
- `최종 반영(Finalize)` 후 `/infer`에 `run_id`를 주지 않으면, **현재 반영된(LIVE) 운영 모델**이 사용됩니다.
- `/playground`는 `[LIVE]` RUN을 기본 선택하지만, 드롭다운에서 다른 RUN으로 바꿔 비교 테스트할 수 있습니다.

6. 평가
- `AUTO_SPLIT=1`(기본): 학습과 **동일한 20% 홀드아웃**으로 평가합니다.

python3 -m sllm.train.evaluate --train_file sample_data/train.jsonl --auto_split --train_ratio 0.8 --tokenizer_dir artifacts/tokenizer --model_dir artifacts/model_dev

- 별도 검증 파일:

python3 -m sllm.train.evaluate --valid_file sample_data/valid.jsonl --tokenizer_dir artifacts/tokenizer --model_dir artifacts/model_dev

이어학습/재개 모드(토크나이저/학습/평가 공통):
- `--resume auto` (`make` 기본값): 중단 지점/체크포인트가 있으면 이어서 진행
- `--resume always`: 반드시 이어서 진행, 재개 파일이 없으면 에러
- `--resume never`: 항상 처음부터 다시 시작

속도 최적화(품질 유지 목적):
- 대규모 학습 데이터(5,000건 이상)에서는 학습 중 저장/평가를 `epoch` 단위로 자동 전환
- CUDA 사용 시 bf16(지원 시)/fp16, TF32, DataLoader worker를 자동 적용
- 평가는 배치 생성(`--batch_size`, 기본 16)으로 처리

디바이스 선택(학습/평가 공통):
- `--device auto` (기본값): CUDA -> MPS -> CPU 순 자동 선택
- `--device cpu`: CPU 강제
- `--device gpu`: GPU(CUDA/MPS) 강제, 없으면 에러

예시:
python3 -m sllm.tokenizer.train_tokenizer --train_file sample_data/train.jsonl --auto_valid_ratio 0.2 --output_dir artifacts/tokenizer --vocab_size 4000 --resume auto
python3 -m sllm.train.train_decoder --train_file sample_data/train.jsonl --tokenizer_dir artifacts/tokenizer --config_file configs/model_dev.yaml --output_dir artifacts/model_dev --auto_split --train_ratio 0.8 --device cpu --resume auto
python3 -m sllm.train.evaluate --train_file sample_data/train.jsonl --auto_split --train_ratio 0.8 --tokenizer_dir artifacts/tokenizer --model_dir artifacts/model_dev --device gpu --resume auto --batch_size 32

학습 또는 평가가 끝나면 `artifacts/model_dev/training_result.json` 파일이 자동 생성/갱신됩니다.

원할 때 수동으로 다시 생성:
python3 -m sllm.train.report_results --model_dir artifacts/model_dev

infer_input.json 기준 message 검색 + 결과(command, reason, accuracy) 확인(`AUTO_SPLIT=1`과 맞추려면 학습과 동일 분할):
python3 -m sllm.train.report_result --model_dir artifacts/model_dev --input_file sample_data/infer_input.json --train_file sample_data/train.jsonl --auto_split --train_ratio 0.8

학습/검증 데이터 포맷(legacy_input 제거):
{
  "record_id": "40004",
  "domain": "oms",
  "system": "OMS",
  "message": "User with id admin access to ip 10.0.0.225 and checked the/dw/main/mainPageSD",
  "create_date": "2026-04-09 11:04:19",
  "state": "PAGE",
  "command": "USER_BY_LOGIN",
  "reason": "access status by user id"
}

주의:
- `record_id`는 **선택(optional)** 입니다. (있으면 추적/중복제거에 유용)
- `command`는 **라벨(label)** 입니다. 학습용(Gold/Silver) 데이터에는 있어야 하지만, Bronze/무라벨 JSONL에는 없어도 됩니다.
- API `/infer` 입력에는 `command`가 **필수가 아니며**, `command`/`reason`/`accuracy`를 보내더라도 무시됩니다.

데이터 3계층 운영 예시:
- `sample_data/gold.jsonl`: 사람 검증 완료 라벨 데이터(학습 핵심)
- `sample_data/silver.jsonl`: 자동 라벨 데이터(`auto_label_source`, `auto_label_confidence` 포함)
- `sample_data/bronze.jsonl`: 무라벨 원천 데이터

현재 샘플 기본 학습 세트:
- `sample_data/train.jsonl`: Gold + Silver에서 학습용으로 선별한 라벨 데이터
- `sample_data/valid.jsonl`: Gold 홀드아웃 검증 데이터

Bronze -> Silver 자동 라벨링(규칙 기반):
python3 -m sllm.train.bronze_to_silver --bronze_file sample_data/bronze.jsonl --silver_file sample_data/silver.jsonl --append --min_confidence 0.8

옵션:
- `--dry_run`: 파일 저장 없이 결과 건수만 확인
- `--reject_file sample_data/rejects.jsonl`: 라벨링 실패/저신뢰 샘플 저장
- `--max_rows N`: 앞에서 N건만 테스트

학습 데이터 분포 시각화(HTML 리포트, 80/20 자동 분할 통계):
python3 -m sllm.train.visualize_data_distribution --train_file train_data/train.jsonl --auto_split --train_ratio 0.8 --output_dir artifacts/model_dev

reason 누락률 점검(JSONL 입력 기준):
python3 -m sllm.train.check_reason_coverage --model_dir artifacts/model_dev --input_file train_data/valid.jsonl

train.jsonl에서 reason 없는 행 제거:
python3 -m sllm.train.filter_missing_reason --input_file train_data/train.jsonl --output_file train_data/train_no_missing_reason.jsonl

원본 덮어쓰기:
python3 -m sllm.train.filter_missing_reason --input_file train_data/train.jsonl --in_place

7. API 실행
- 권장(단축 명령):
  - `make api`

- 직접 실행:
  - `PYTHONPATH=src uvicorn sllm.api.server:app --host 0.0.0.0 --port 8000`

환경 변수:
- `SLLM_MODEL_DIR` (기본: `artifacts/model_dev`)
- `SLLM_TOKENIZER_DIR` (기본: `SLLM_MODEL_DIR`)
- `SLLM_COMMANDS_FILE` (기본: `<MODEL_DIR>/commands.json`)
- `SLLM_EXPERIMENTS_DIR` (기본: `artifacts/experiments`, RUN_ID 실험 탐색 루트)

Command 관리(브라우저):
- `http://127.0.0.1:8000/commands`
- 여기서 command 목록을 추가/수정/삭제 가능
- 예측 출력 command는 이 목록 중 하나로 정규화됨
- reason은 command 정규화 후에도 입력 message 의도/엔티티를 반영해 동적으로 생성됨(정적 템플릿 우선 아님)

기본 8개 command:
- `USER_BY_LOGIN`
- `TOP_IP_BY_PAGE`
- `DELIVERY_VEHICLE`
- `RESYNC_ORDER`
- `UPDATE_LOCATION`
- `SYNC_INVENTORY`
- `RESCHEDULE_SHIPMENT`
- `NO_ACTION`
- 참고 파일: `sample_data/commands_default.json`

8. 상태 확인
curl http://127.0.0.1:8000/health

브라우저 화면:
- 대시보드: `http://127.0.0.1:8000/dashboard` (**최종 반영된 LIVE 모델 기준**)
- 학습 업로드/실행: `http://127.0.0.1:8000/train`
- 5단계 마법사: `http://127.0.0.1:8000/wizard`
  - 1) 학습
    - `run_id` 입력 기본값: `run_YYYYMMDD_HHMMSS` (예: `run_20260414_153045`)
  - 2) 학습결과 테스트
  - 3) 반영
  - 4) 반영 후 테스트
  - 5) 대시보드 최종 반영 확인
  - RUN 선택 영역의 `선택 RUN 모델정보(팝업)` 버튼으로, 반영 전 테스트용 학습 RUN의 상세 정보를 팝업에서 확인 가능
  - RUN 선택 영역의 `선택 RUN 모델 삭제` 버튼으로, 선택한 RUN의 실험 모델(`artifacts/experiments/<run_id>`) 삭제 가능
    - 보호 규칙: `__default__` 및 현재 LIVE `run_id`는 삭제 불가
    - 삭제 시 업로드 샘플 흔적(`data/samples/<run_id>`, `data/samples/<run_id>_*.jsonl`)과 해당 RUN job 이력도 함께 정리
- 최종 반영 전용 화면: `http://127.0.0.1:8000/release` (검수 승인 → dry-run → 최종 반영)
- 추론 확인: `http://127.0.0.1:8000/playground` (RUN_ID 선택 가능)
- 오타 호환: `http://127.0.0.1:8000/playgroud` 로 접속해도 `/playground`로 리다이렉트됩니다.
- 학습 결과: `http://127.0.0.1:8000/results` (RUN_ID 선택 + 2개 RUN 비교 + 리포트 확인)
- 결과 비교 URL 예시: `http://127.0.0.1:8000/results?run_id=run_A&compare_run_id=run_B`

대시보드에 표시되는 항목(LIVE 기준):
- 최종 반영된 `run_id` (current_run_id)
- 승격(promotion) 정보(승격 시각, source/target 경로)
- 모델 구성(config/run_meta), 아티팩트 존재/mtime/size
- 학습 상태(최근 job 상태, running/queued/failed 카운트, 최근 20개 job)
- 품질 게이트/Reason 커버리지/데이터 분포 리포트

9. 추론 요청
curl -X POST "http://127.0.0.1:8000/infer" -H "Content-Type: application/json" -d "{\"system\":\"WMS\",\"message\":\"picking location not found\",\"warehouse_id\":\"WH01\"}"

특정 실험(RUN_ID) 모델로 추론:
curl -X POST "http://127.0.0.1:8000/infer?run_id=run_20260414_103000" -H "Content-Type: application/json" -d "{\"system\":\"WMS\",\"message\":\"picking location not found\",\"warehouse_id\":\"WH01\"}"

중요:
- 입력 JSON에는 `command`/`reason`/`accuracy`를 넣지 않습니다. (`system`, `message` 등 입력 정보만 전달)
- 출력 JSON에만 `command`, `reason`, `accuracy`가 반환됩니다.
- 생성/파싱 실패 시 fail-safe 응답을 반환합니다:
  - `command`: `NO_ACTION`
  - `reason`: `model output could not be parsed safely`
  - `accuracy.command_accuracy`: `null`

---

모델 품질(정확도/Reason) 개선 관련 소스 변경 사항 (2026-04-14)
- **입력 직렬화 개선**: 학습/추론 프롬프트의 입력을 `k=v | ...` 텍스트 대신 **JSON 문자열(`INPUT_RECORD_JSON`)**로 제공합니다. 특수문자/경로/IP 등 레거시 입력이 더 안정적으로 보존되어 학습 신호가 좋아집니다. (`src/sllm/common/prompting.py`, `src/sllm/common/io.py`)
- **Reason 품질 목표 명시**: 프롬프트에 `reason`을 입력 근거로 충분히 설명하도록(가능하면 ~100자+) 제약을 추가했습니다. (학습/추론 프롬프트 공통)
- **평가 정확도 계산 개선**: 서비스 `/infer`는 예측을 **command catalog로 정규화**한 뒤 결과를 반환합니다. 기존 평가는 정규화 없이 raw command만 비교해 실제 서비스 품질과 괴리가 있었으므로, 평가에서도 동일하게 catalog 정규화를 적용해 `command_accuracy`가 운영 품질을 반영하도록 맞췄습니다. (`src/sllm/train/evaluate.py`)
- `run_id`가 없으면 기본 모델(`SLLM_MODEL_DIR`)을 사용합니다.

NO_ACTION(fail-safe) 진단/완화:
- 추론 시 원문 생성/파싱 정보를 `infer_debug_last.json`(최신 1건), `infer_debug_history.jsonl`(누적)에 자동 저장합니다.
- 파싱이 완전히 실패해도 JSON 유사 출력에서 `command`/`reason`를 복구 시도합니다.
- 생성 반복/깨짐 완화를 위해 추론 기본값은 deterministic + 반복 억제 튜닝이 적용됩니다.
- 튜닝 환경변수:
  - `SLLM_MAX_NEW_TOKENS` (기본 `96`)
  - `SLLM_REPETITION_PENALTY` (기본 `1.08`)
  - `SLLM_NO_REPEAT_NGRAM_SIZE` (기본 `4`)
  - `SLLM_SAVE_INFER_DEBUG` (기본 `1`)
  - `SLLM_APPEND_INFER_DEBUG` (기본 `1`)

검수 승인 API 예시:
curl -X POST "http://127.0.0.1:8000/runs/review/approve" -H "Content-Type: application/json" -d "{\"run_id\":\"run_20260414_103000\",\"reviewer\":\"rms\",\"note\":\"품질/정확도 확인 완료\",\"approved\":true}"

최종 반영 dry-run API 예시:
curl -X POST "http://127.0.0.1:8000/runs/finalize" -H "Content-Type: application/json" -d "{\"run_id\":\"run_20260414_103000\",\"dry_run\":true,\"min_command_accuracy\":0.82,\"min_reason_coverage\":1.0,\"max_command_malformed_ratio\":0.0,\"require_review_approval\":true}"

최종 반영 후 실제 운영 API 출력 확인:
- playground: `http://127.0.0.1:8000/playground` 에서 `[LIVE]` RUN 선택 후 테스트
- 선택 RUN 바로 테스트: `http://127.0.0.1:8000/playground?run_id=<RUN_ID>`
- 실제 API: `curl -X POST "http://127.0.0.1:8000/infer" -H "Content-Type: application/json" -d "{\"system\":\"WMS\",\"message\":\"picking location not found\"}"`

10. 입력 JSON 파일로 로컬 추론 결과 확인
python -m sllm.infer.run_infer_json --model_dir artifacts/model_dev

처음 실행하면 `sample_data/infer_input.json` 템플릿이 생성됩니다.
추론 결과는 `artifacts/model_dev/infer_result.json`에 저장됩니다.
검증 비교 결과는 `artifacts/model_dev/infer_compare.json`에 저장됩니다.
비교 시 검증 데이터 검색은 입력의 `message`를 최우선 기준으로 수행합니다.
- RUN_ID 실험 모델이면 `--model_dir artifacts/experiments/<RUN_ID>/model`를 사용합니다.

11. 지속 고품질 운영 루틴(추천)
우선 권장: `http://127.0.0.1:8000/wizard`에서 1~5단계를 순서대로 수행

1) 새 배치 데이터 준비  
2) `data_quality_gate` 실행 (`PASS/FAIL`, 필수 컬럼/중복/충돌/분포 경고 확인)  
3) `FAIL`이면 데이터 정제 후 재실행  
4) `RUN_ID`를 새로 발급해 실험 분리 학습  
5) `/results`에서 기존 RUN과 비교(정확도/샘플 수/품질 상태)  
6) `/release`에서 `검수 승인` 저장  
7) `/release`에서 `DRY_RUN 검증` 실행  
8) `/release`에서 `최종 반영` 실행(기본 모델 경로로 promote)  
9) `/playground`에서 `[LIVE]` RUN으로 최종 반영 결과 테스트  
10) 실제 API 출력은 `/infer`(run_id 없이 기본 운영 모델 호출)로 검증

혼동 방지(중요):
- 학습 명령 예: `make train TRAIN_FILE=train_data/incheon_backend_error_only_train.jsonl RESUME=never RUN_ID=run-20260413-120300`
- 최종 반영 명령 예: `make finalize FINALIZE_RUN_ID=run-20260413-120300`
- `make finalize` 실행 시 보통 `RUN_ID`는 주지 않습니다. (`RUN_ID`를 같이 주면 타깃 경로가 실험 경로로 바뀔 수 있음)

`data_quality_report.json` 자동 항목:
- 필수 컬럼 검증: `required_columns.missing_or_blank_counts`
- 중복 체크: `duplicates.exact_duplicate_rows`, `duplicates.duplicate_input_same_label`
- 충돌 체크: `conflicts.input_label_conflicts` (+ 샘플)
- 분포 편향 경고: `warnings` (command/system/domain 편향 경고)
- 통과/실패: `status` + `fail_reasons`

실험 RUN 조회 API:
- `GET /runs/list` (playground/results에서 동일 사용)
- `GET /runs/detail?run_id=<RUN_ID>` (해당 RUN의 학습/품질/분포/파일/잡 이력 상세)
- `POST /runs/delete` (선택 RUN 실험 모델 삭제 + 샘플/잡 흔적 정리)
- `GET /runs/review?run_id=<RUN_ID>` (검수 승인 상태 조회)
- `POST /runs/review/approve` (검수 승인/해제 저장)
- `GET /runs/current` (현재 최종 반영된 RUN_ID 확인)
- `POST /runs/finalize` (품질/정확도 조건 검증 후 최종 반영 실행)

RUN_ID가 `/playground`에 안 보일 때 체크:
1) 학습 산출물이 `artifacts/experiments/<run_id>/model` 아래 생성되었는지 확인  
2) 업로드만 되고 학습 실패한 경우엔 model artifact가 없을 수 있음(이 경우 `/train` job 상태 확인)  
3) 서버 재시작 직후엔 메모리 job 이력이 초기화될 수 있으므로, 최종 기준은 `artifacts/experiments` 실제 폴더 구조
4) 구버전 업로드 형식(`data/samples/<run_id>_*.jsonl`)도 RUN 후보로 표시되지만, model artifact가 없으면 `exists=false`로 보입니다.
5) `/wizard`의 RUN 선택은 가능한 RUN을 전체 노출합니다.  
   - `/runs/list` 호출이 실패해도 드롭다운이 비지 않도록 `default (fallback)` 옵션을 자동 표시합니다.
   - `/wizard` 최초 렌더링 시 서버사이드로 RUN 옵션을 미리 채워, JS 오류/네트워크 문제에서도 RUN 목록이 보이도록 보강했습니다.
   - RUN_ID 선택 섹션 버튼(목록 새로고침/결과 보기/playground/모델정보/삭제)이 동작하지 않던 이슈를 수정했습니다.
     (`watchTrainingJobSSE` 로그 누적 문자열의 개행 이스케이프 오류로 JS 파싱이 중단되던 문제)
6) `/playground`의 RUN 선택은 기본적으로 `exists=true` 또는 `[LIVE]` RUN만 표시합니다.
7) `/wizard` 4단계 `LIVE 테스트 실행`에서 HTTP 500이 나던 문제를 보강했습니다.
   - 원인: LIVE 기본 모델(`artifacts/model_dev`)의 `tokenizer.json` 또는 모델 파일이 없을 때 `/infer` 엔진 로드 예외가 그대로 전파됨
   - 현재: `/infer`는 엔진 로드 실패 시에도 500 대신 fail-safe JSON(`NO_ACTION`)을 반환하며, `reason`에 로드 실패 원인을 포함해 화면에서 바로 확인할 수 있습니다.
8) `/wizard` 1단계 학습 모니터 UI를 강화했습니다.
   - `학습 시작` 버튼 하단 로그 패널이 길어져도 화면이 깨지지 않도록 세로 스크롤(`max-height + overflow`)을 적용했습니다.
   - 로그를 `토크나이저 / 모델학습 / 평가 / 전체`로 분리해 확인할 수 있게 개선했습니다.
   - 실시간 지표 카드와 최근 로그 테이블을 추가해 `loss`, `grad_norm`, `learning_rate`, `epoch`를 모니터링할 수 있습니다.
   - 진행률 로그(`xx% | current/total`)를 별도 표시해 학습 진행 상태를 빠르게 확인할 수 있습니다.
9) `/infer` 안정성 가드를 추가 보강했습니다.
   - 엔진 조회 함수(`_get_engine_for_run`)에서 예외가 발생해도 API가 500으로 터지지 않도록 `/infer`에서 한 번 더 예외를 포착합니다.
   - 현재 동작은 항상 fail-safe JSON(`NO_ACTION`)을 반환하며, `reason`에 로드 실패 원인을 포함합니다.
   - 기존 uvicorn 프로세스가 구버전 코드를 들고 있으면 동일 traceback이 계속 보일 수 있으므로 서버 재시작 후 확인합니다.
10) Wizard 단계 이동/반영 플로우를 사용자 요청에 맞게 조정했습니다.
   - `2단계/3단계/4단계/5단계 이동` 버튼은 항상 클릭 가능하도록 변경했습니다. (완료 여부는 배지로 표시)
   - 4단계 `반영 후 테스트`는 성공 시 자동으로 5단계로 점프하지 않고, 사용자가 `5단계로 이동` 버튼을 눌러 진행합니다.
   - `5단계로 이동` 버튼 클릭 시 선택 RUN이 아직 LIVE가 아니면, Wizard가 `최종 반영`을 자동 실행해 LIVE 반영을 먼저 시도합니다.
   - 자동 반영 실패 시(검수 승인/품질 조건 미충족 등) 5단계 이동을 막고 안내 메시지를 표시합니다.
   - 실패 안내 메시지에는 finalize 검증 오류(`validation.errors`)를 그대로 표시해, 어떤 조건 때문에 반영이 막혔는지 바로 확인할 수 있습니다.
   - 검수 승인 누락만 원인인 경우, Wizard가 `reviewer=wizard-auto`로 자동 승인 저장 후 최종 반영을 1회 재시도합니다.
11) Wizard 5단계 자동 보정(리포트 누락 대응)을 추가했습니다.
   - `data_quality_report.json`, `reason_coverage_report.json` 누락으로 finalize가 막히면 `/runs/prepare_finalize`를 호출해 해당 리포트를 자동 생성한 뒤 반영을 재시도합니다.
   - 입력 데이터 경로는 아래 순서로 자동 탐색합니다.
     `run_meta.input_source` → `data_quality_report.input_source` → `data_distribution_report(train_file/valid_file)` → `eval_results.jsonl(valid_file 태그)` → run job 이력 → `data/samples/<run_id>`
   - `eval_results.jsonl`의 `auto_split:<train_source>:<ratio>` 태그를 찾으면 `reason_coverage` 계산도 동일 `train_ratio`로 자동 실행합니다.
   - 보정 실패 시에도 실패 상세 원인을 알림으로 그대로 보여줍니다.
12) Wizard 4단계 LIVE 테스트 자동 복구를 추가했습니다.
   - 4단계 `/infer` 결과가 `NO_ACTION` + `run_id model load failed: __default__`(또는 `model not found`)이면, 선택 RUN을 자동 반영 시도한 뒤 LIVE 테스트를 1회 자동 재실행합니다.
   - 자동 반영 후에도 선택 RUN과 현재 LIVE RUN이 일치하지 않으면, 무반응 대신 상태 불일치 상세 알림을 표시합니다.

최종 반영 CLI:
python3 -m sllm.train.finalize_run --run_id run_20260414_103000 --target_model_dir artifacts/model_dev --target_tokenizer_dir artifacts/model_dev --min_command_accuracy 0.82 --min_reason_coverage 1.0 --max_command_malformed_ratio 0.0

검증만 수행(dry-run):
python3 -m sllm.train.finalize_run --run_id run_20260414_103000 --dry_run

검수 승인 없이 반영(비권장, 긴급 시):
python3 -m sllm.train.finalize_run --run_id run_20260414_103000 --skip_review_approval



-----------
결과 확인 

실행 방법:
PYTHONPATH=src python3 -m sllm.infer.run_infer_json --model_dir artifacts/model_dev

동작:
1. sample_data/infer_input.json이 없으면 자동 생성
2. 입력 JSON 읽어서 추론 수행
3. 결과 JSON을 artifacts/model_dev/infer_result.json에 저장
* 원하면 --input_file, --output_file로 경로를 바꿔서 쓸 수 있습니다.
* 

-----------
Makefile 사용 예시

- 데이터 품질 게이트:
  - `make quality TRAIN_FILE=sample_data/train.jsonl MODEL_DIR=artifacts/model_dev`

- 토크나이저:
  - `make tok TRAIN_FILE=sample_data/train.jsonl TOKENIZER_DIR=artifacts/tokenizer`

- 학습 + 자동 평가(토크나이저 → 학습 80/20 → evaluate):
  - `make train TRAIN_FILE=sample_data/train.jsonl MODEL_DIR=artifacts/model_dev`

- RUN_ID 실험 분리 학습(권장):
  - `make train TRAIN_FILE=sample_data/train.jsonl RUN_ID=run_20260414_103000`

- 학습(별도 valid 사용):
  - `make train TRAIN_FILE=sample_data/train.jsonl VALID_FILE=sample_data/valid.jsonl AUTO_SPLIT=0`

- 평가(기본은 `TRAIN_FILE` 홀드아웃 20%):
  - `make eval TRAIN_FILE=sample_data/train.jsonl`

- API:
  - `make api PORT=8000`

- 최종 반영(운영 모델 갱신):
  - `make finalize FINALIZE_RUN_ID=run_20260414_103000 MODEL_DIR=artifacts/model_dev TOKENIZER_DIR=artifacts/model_dev MIN_COMMAND_ACCURACY=0.82`

- 최종 반영 검증만(dry-run):
  - `make finalize FINALIZE_RUN_ID=run_20260414_103000 DRY_RUN=1`

- 검수 승인 없이 최종 반영(비권장):
  - `make finalize FINALIZE_RUN_ID=run_20260414_103000 REQUIRE_REVIEW_APPROVAL=0`

-----------
브라우저에서 학습 실행(업로드)

- `http://127.0.0.1:8000/train`에서 `.jsonl` 파일을 업로드하면 백그라운드에서 다음 형태로 실행됩니다:
  - `python3 scripts/run_detached_train_job.py --meta_path artifacts/jobs/train_<job_id>.json --log_path artifacts/jobs/train_<job_id>.log --workdir <repo_root> --command ".venv/bin/python scripts/train.py --input data/samples/<run_id>/<업로드파일>.jsonl --run_id <run_id>"`
  - (macOS에서는 학습 중 절전 방지를 위해 내부적으로 `caffeinate -dimsu`가 커맨드 앞에 붙을 수 있습니다.)
  - 업로드 파일명에 공백/특수문자/한글이 포함돼도 학습 인자 깨짐이 없도록 실행 인자를 shell-quote 처리합니다.
  - 학습 Python 실행 경로는 우선순위로 선택됩니다: `SLLM_TRAIN_PYTHON` → `<repo>/.venv/bin/python`(또는 Windows `Scripts/python.exe`) → 현재 서버 Python(`sys.executable`).
  - 큰 파일 업로드 안정성을 위해 서버는 업로드를 스트리밍(청크)으로 저장합니다. (`/train/upload`가 파일 전체를 한 번에 메모리 로드하지 않음)
- 업로드 화면에서 `run_id`를 지정할 수 있고, 비우면 자동 생성됩니다.
- `/train` 목록에서 `run_id`별로 `/results?run_id=<run_id>` 링크로 바로 이동할 수 있습니다.

견고한 학습 실행(브라우저/서버 종료와 무관):
- 학습은 서버와 분리된 **detached runner 프로세스**로 실행되므로, 브라우저를 닫거나 uvicorn을 재시작해도 학습이 계속됩니다.
- 잡 메타는 `artifacts/jobs/train_<job_id>.json`, 로그는 `artifacts/jobs/train_<job_id>.log`에 저장됩니다.
- UI는 SSE(`/train/jobs/<job_id>/events`)로 상태/로그를 스트리밍하며, 재접속 시 자동으로 이어서 표시합니다.
- 최종 반영은 `/release`에서 `검수 승인` → `DRY_RUN` → `최종 반영` 순서로 수행합니다.
- 가장 간단한 운영 흐름은 `/wizard` 한 화면에서 1) 학습 2) 학습결과 테스트 3) 반영 4) 반영 후 테스트 5) 대시보드 확인 순으로 진행하는 방식입니다.

- 필요 패키지:
  - 파일 업로드(FormData)를 위해 `python-multipart`가 필요합니다. (`requirements.txt`에 포함)
