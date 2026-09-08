# Robust VLA Recovery

LIBERO 시뮬레이션에서 VLA 정책의 실패 징후를 시간축으로 감지하고, 제한된 복구
행동으로 조작 성공률을 높일 수 있는지 검증하는 연구 프로젝트다. 정책 실행부터
실패 감지, 복구, OOD 평가까지 같은 기록 형식과 평가 조건으로 연결하는 것이 목표다.

## 연구 질문

> 실패 감지와 제한된 규칙 기반 복구가 교란 환경에서 VLA 로봇 조작의 최종
> 성공률을 높이는가?

정상 환경의 정책, 교란 데이터로 학습한 정책, 실패 감지기를 붙인 정책, 감지 후
복구까지 수행하는 정책을 같은 조건에서 비교한다. 평균 성공률뿐 아니라 오탐 복구,
해로운 복구, 실패 감지 선행 시간과 제어 지연시간도 함께 보고한다.

## 시스템 구성

```text
언어 + 이미지 + 로봇 상태
  -> VLA 정책
  -> 행동
  -> LIBERO 환경
  -> 시간축 실패 감지기
  -> 복구 관리자
  -> 수정된 행동
```

실패 감지기는 최근 K개 시점의 시각 특성, 로봇 상태와 행동을 사용한다. 복구
관리자는 `reobserve`, `backoff`, `regrasp`, `retry_subtask`, `abort` 중 하나를
선택하는 규칙 기반 상태 기계로 설계한다.

## 조작 태스크

| 태스크 | 목표 | 대표 실패 |
|---|---|---|
| `pick_place` | alphabet soup를 basket 안에 넣기 | 잡기 실패, 다른 객체 선택, 떨어뜨림 |
| `stack` | 앞쪽 검은 bowl을 가운데 bowl 위에 쌓기 | 정렬 실패, 떨어뜨림, 정지 |
| `open_drawer` | cabinet의 위쪽 drawer 열기 | 다른 손잡이 선택, 정지, 충돌 |
| `shelf_place` | 가운데 book을 cabinet shelf에 놓기 | 잡기 실패, 떨어뜨림, 충돌 |

모든 태스크는 `libero_90`에서 실행하며 최대 400 스텝, 제어 주기 20 Hz를 사용한다.
성공 판정은 LIBERO의 기본 성공 조건을 우선하고, 물체가 놓인 상태가 10 스텝 연속
유지되는지도 별도 진단값으로 기록한다.

## 비교 실험

| 비교군 | 학습 데이터 | 실패 감지 | 복구 | 확인할 내용 |
|---|---|---|---|---|
| A `Base` | 정상 | 없음 | 없음 | 정상 정책의 최소 기준선 |
| B `Augmentation` | 정상 + 교란 | 없음 | 없음 | 교란 데이터 학습의 효과 |
| C `Detector` | 정상 + 교란 | 있음 | 기록만 | 감지 품질과 오탐 비용 |
| D `Recover` | 정상 + 교란 | 있음 | 있음 | 폐루프 복구의 순기여 |

핵심 비교는 같은 강건 정책을 쓰는 B와 D다. 두 비교군의 차이로 감지기와 복구
시스템이 성공률에 기여한 정도를 측정한다.

## 주요 평가 지표

- 정상 및 OOD 태스크 성공률과 성능 하락 폭
- 실패 유형 Macro-F1과 AUROC
- 실패 시작 대비 감지 선행 시간
- 정상 에피소드의 오탐 횟수와 오탐 복구율
- 복구율과 해로운 복구율
- 태스크 시간, 재시도 횟수, 정책·감지기·관리자의 처리 지연시간

## 현재 구현 범위

현재 저장소에는 네 태스크 실행, 무손실 HDF5 기록, 기준선 정책 연결과 정상 조건
평가 기능이 구현돼 있다. 현재 기준선은 입출력 연결을 확인하기 위한 미학습 경량
정책이며, 학습된 VLA 성능을 나타내지 않는다. 다음 단계에서는 공식 시연 데이터를
검사하고 정책 학습용 형식으로 변환한다.

일차별 증거, 재검증 결과와 예외 사항은 README가 아니라
[프로젝트 계약서의 1주차 검증 기록](docs/project_charter.md#13-1주차-검증-기록)에
보관한다.

## 시드 원칙

새 단일 시드 실행의 기본값은 `378`이다. 이미 검증한 과거 산출물은 당시 시드를
그대로 유지한다. `configs/rollout.toml`도 Day 5의 다섯 시드 기준 자료를 보존한다.

## 저장소 구성

```text
configs/        실행 설정과 태스크 정의
src/            설치 가능한 Python 패키지
scripts/        실제 시뮬레이터 계약 검사
tests/          단위 검사와 기본 동작 검사
docs/           설계 결정과 일차별 평가 기록
outputs/        실행 산출물, Git 추적 제외
```

구현이 없는 미래 기능용 빈 패키지와 빈 디렉터리는 두지 않는다. 해당 기능을
구현할 때 필요한 위치만 추가한다.

## 핵심 패키지 설치와 검사

`rvla-*` 명령을 사용하기 전에 패키지를 설치한다. 핵심 패키지는 외부 실행
의존성이 없으며 Python 3.12를 지원한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

## 시뮬레이션 환경

LeRobot 0.6.1의 LIBERO 통합은 Linux 전용이다. Windows에서는 WSL2의 Ubuntu
24.04 환경을 사용한다.

```bash
conda env create -f environment.yml
conda activate robust-vla-recovery
python -m pip install uv==0.12.9
uv pip sync requirements/simulation-linux.lock --torch-backend cu128 --no-build-isolation
uv pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

직접 의존성은 `requirements/simulation-linux.txt`에 적혀 있다. 생성된
`requirements/simulation-linux.lock`은 Linux와 CUDA 12.8 환경에서 사용하는
전이 의존성 155개의 버전을 고정한다.

## LIBERO 실행 확인

두 카메라와 EGL 화면 밖 렌더링을 사용해 고정 시드와 초기 상태로 에피소드 하나를
실행한다.

```bash
conda activate robust-vla-recovery
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  rvla-run-libero-smoke configs/tasks/libero_smoke.toml
```

실행할 때마다 `outputs/day03/` 아래에 고유한 디렉터리가 생기며 `episode.mp4`,
`trajectory.npz`, `first_frame.png`, `metadata.json`을 저장한다. 화면 서버 없이
저장 결과를 검증하려면 다음 명령을 사용한다.

```bash
rvla-replay-episode outputs/day03/<run>/metadata.json
```

재생기는 파일 해시 불일치, 궤적 길이 오류, 영상 프레임 누락을 거부한다.
실행 근거와 해석은 [LIBERO 실행 평가](docs/day03_evaluation.md)에 기록했다.

## 태스크 어댑터

```bash
rvla-run-tasks --list
rvla-run-tasks --task all --seed 378
rvla-run-tasks --task stack --seed 378 --instruction-index 2
```

`--task`에는 `pick_place`, `stack`, `open_drawer`, `shelf_place`, `all`을 넣을 수
있다. `configs/tasks/catalog.toml`은 태스크의 고유 이름, 목표 물체, 성공 조건,
400 스텝 제한, 20 Hz 제어 주기를 고정한다. 각 태스크에는 뜻이 같은 지시문 5개가
있다. CLI는 설치된 BDDL의 목표와 언어가 태스크 목록과 다르면 실행을 중단한다.

각 실행은 영상, 메타데이터, 궤적을 `outputs/day04/`에 저장하고 영상 전체와
체크섬을 검증한다. 별도 시뮬레이터 계약 검사는 다음과 같이 실행한다.

```bash
python scripts/check_task_adapters.py
```

공식 성공은 LIBERO 기본 성공 조건이 참이 되는 즉시 기록한다. 같은 400 스텝
예산 안에서 성공과 물체 해제가 10 스텝 연속 유지되는지도 진단한다. 자세한
판정 규칙과 실제 결과는 [태스크 어댑터 평가](docs/day04_evaluation.md)에 있다.

## 무손실 실행 기록

로거는 두 카메라의 원본 RGB, 로봇 상태, 행동, 시점별 언어, 보상, 성공과
종료 신호를 HDF5에 연속 기록한다. 행동 T개와 관측 T+1개를 정확한 시뮬레이션
시간과 함께 보존하며, RGB는 gzip으로 무손실 압축한다.

```bash
rvla-rollouts collect --dry-run
rvla-rollouts collect --config configs/rollout.toml
rvla-rollouts verify outputs/day05/<batch>/batch.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

기본 수집 설정은 네 태스크와 시드 다섯 개를 조합한 400 스텝 에피소드 20개다.
검증기는 모든 RGB 프레임을 풀어 수집 당시 픽셀 해시와 비교하고, 길이와 시간 정렬,
완료 상태를 확인한다. 형식은 [실행 기록 저장 형식](docs/rollout_format.md), 실측 결과는
[실행 기록 평가](docs/day05_evaluation.md)에 기록했다.

GPU 없이 데이터 계층만 검사하려면 다음 의존성을 설치한다.

```bash
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## 기준선 정책 연결

진단 정책은 두 RGB 카메라, 34차원 로봇 상태와 언어를 받아 범위가 제한된
7차원 행동을 출력한다. 가중치는 시드 378로 고정했지만 학습되지 않았으므로
이 실행은 입출력과 처리 시간만 검사한다.

```bash
rvla-policy-rollout run --config configs/policy.toml --dry-run
rvla-policy-rollout run --config configs/policy.toml
rvla-policy-rollout review outputs/day06/<run>/metadata.json \
  --candidate stalled --frame 0 --frame 100 --note "수동 검토 메모"
```

측정값과 한계는 [기준선 정책 평가](docs/day06_evaluation.md)에 있다.

## 정상 조건 기준선 평가

평가기는 네 태스크를 시드 378부터 387까지 실행한다. 에피소드 시간, 추론 지연시간,
CUDA 메모리를 기록하고 실패 영상 5개만 보존한다. 검증 단계에서는 평가 행렬,
사람이 남긴 실패 검토, 영상 SHA-256을 다시 확인한다.

```bash
rvla-baseline-eval run --config configs/baseline_eval.toml --dry-run
rvla-baseline-eval run --config configs/baseline_eval.toml
rvla-baseline-eval review outputs/day07/<run>/results.json \
  --episode <episode-id> --candidate stalled --frame 0 --note "검토 메모"
rvla-baseline-eval verify outputs/day07/<run>/results.json
```

태스크별 결과와 실패 원인, 다음 데이터 수집 계획은
[정상 조건 기준선 평가](docs/day07_evaluation.md)에 기록한다.
