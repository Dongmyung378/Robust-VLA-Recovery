# Robust VLA Recovery

LIBERO에서 시간축 실패 감지와 폐루프 복구를 연구하기 위한 VLA 로봇 조작
프로젝트다. 현재 1주차인 Day 1부터 Day 7까지 완료했으며, 네 태스크의 실행,
무손실 기록, 기준선 평가가 자동화돼 있다.

## 현재 진행 상황

- Day 1: 연구 질문, 태스크, 지표, 자원 예산과 범위 축소 순서를 확정했다.
- Day 2: 저장소 구조, 재현 환경, 공통 실행 설정, 검증기, 기본 테스트와 CI를 만들었다.
- Day 3: WSL2와 CUDA 환경에서 LIBERO 에피소드를 실행하고 화면 없이 영상과
  궤적을 저장한 뒤 재생 검증했다.
- Day 4: 네 태스크 어댑터와 지시문 20개, 공통 성공·시간 초과·실패 규칙,
  통합 CLI를 구현하고 실제 시뮬레이터에서 확인했다.
- Day 5: 무손실 스트리밍 로거로 에피소드 20개와 RGB 프레임 16,040개를
  수집하고 독립 검증했다.
- Day 6: 이미지·상태·언어 기준선 정책을 연결하고 행동 크기와 추론 시간을
  기록하면서 LIBERO를 100 스텝 연속 실행했다.
- Day 7: 정상 조건 에피소드 40개를 평가하고 태스크별 결과표, 실패 영상 5개,
  2주차 데이터 수집 규모를 확정했다.

## 1주차 로드맵 검증

2026-09-08에 원본 로드맵과 현재 저장소를 다시 대조하고, 저장된 산출물을 현재
검증기로 읽었다.

| 일차 | 로드맵 완료 기준 | 재검증 근거 | 판정 |
|---|---|---|---|
| Day 1 | 연구 질문, 범위, 지표, 축소 규칙 기록 | `docs/project_charter.md`에 태스크 4개, 비교군 4개, 지표와 자원 예산이 고정돼 있다. | 통과 |
| Day 2 | 새 환경 설치 후 기본 검사 통과 | Python 3.12 환경 고정, 엄격한 설정 검증, 가져오기 검사, GitHub Actions가 있다. | 통과 |
| Day 3 | 고정 시드 영상과 메타데이터 저장 | 280 스텝, 281 프레임, 영상·궤적·첫 프레임의 SHA-256 재검증 통과. | 통과 |
| Day 4 | 하나의 CLI로 네 태스크 실행 | 네 태스크 1,600 스텝과 영상 1,604 프레임 재생 검증 통과. 실제 reset과 성공 조건 검사 기록도 유효하다. | 통과 |
| Day 5 | 에피소드 20개 무손실 저장·재생 | 8,000 스텝, RGB 프레임 16,040개, HDF5 20개의 길이·픽셀·SHA-256 재검증 통과. | 통과 |
| Day 6 | 정책 출력으로 100 스텝 이상 실행 | 시드 378 정책 실행의 100 스텝, 관측 101개, RGB 프레임 202개와 수동 실패 검토 기록 재검증 통과. | 통과 |
| Day 7 | 태스크별 시드 10개 평가와 결과표 생성 | 40회 모두 완료, 실패 영상 5개 검토, 시간·지연시간·GPU 메모리 기록과 결과표 검증 통과. | 통과 |

1주차 통과 기준인 네 태스크 실행 영상, 기준선 결과, 손상 없이 재생되는 실행 기록
파일이 모두 남아 있다. 빈 `assets` 디렉터리는 파일 수를 줄이기 위해 제거했으며,
실제 자산이 생기는 시점에 다시 만든다. Git은 빈 디렉터리를 추적하지 않으므로
기능과 재현성에는 영향이 없다.

현재 WSL2 환경에서 전체 검사 50개가 건너뛴 항목 없이 통과했고 `pip check`도
문제를 찾지 못했다. Day 3과 Day 4 영상 5개, Day 5 HDF5 20개, Day 6 정책 실행,
Day 7 결과 파일과 실패 영상 5개도 각각 현재 검증기로 다시 확인했다.

Day 6과 Day 7의 `lightweight_mlp_v1`은 학습되지 않은 결정적 시스템 기준선이다.
따라서 Day 7 성공률 0%는 실행 오류가 아니라 현재 정책의 실제 하한 성능이다.
학습된 VLA 성능은 2주차 시연 데이터 학습 이후 같은 평가 계약으로 측정한다.

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

## Day 3 LIBERO 실행 확인

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
실측 결과와 해석은 [Day 3 평가](docs/day03_evaluation.md)에 기록했다.

## Day 4 태스크 어댑터

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
판정 규칙과 실제 결과는 [Day 4 평가](docs/day04_evaluation.md)에 있다.

## Day 5 무손실 실행 기록

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

Day 5 기준 계획은 네 태스크와 시드 다섯 개를 조합한 400 스텝 에피소드 20개다.
검증기는 모든 RGB 프레임을 풀어 수집 당시 픽셀 해시와 비교하고, 길이와 시간 정렬,
완료 상태를 확인한다. 형식은 [실행 기록 저장 형식](docs/rollout_format.md), 실측 결과는
[Day 5 평가](docs/day05_evaluation.md)에 기록했다.

GPU 없이 데이터 계층만 검사하려면 다음 의존성을 설치한다.

```bash
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Day 6 기준선 정책 연결

Day 6 진단 정책은 두 RGB 카메라, 34차원 로봇 상태와 언어를 받아 범위가 제한된
7차원 행동을 출력한다. 가중치는 시드 378로 고정했지만 학습되지 않았으므로
이 실행은 입출력과 처리 시간만 검사한다.

```bash
rvla-policy-rollout run --config configs/policy.toml --dry-run
rvla-policy-rollout run --config configs/policy.toml
rvla-policy-rollout review outputs/day06/<run>/metadata.json \
  --candidate stalled --frame 0 --frame 100 --note "수동 검토 메모"
```

기준 실행은 정책 출력으로 100 스텝을 연속 진행했다. 측정값과 한계는
[Day 6 평가](docs/day06_evaluation.md)에 있다.

## Day 7 정상 조건 기준선 평가

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

기준 평가는 40회 모두 완료됐다. 모든 태스크에서 성공 0/10, 평균 추론 지연시간
5.144 ms, 50 ms 초과 0회를 기록했다. 이는 학습된 VLA 성능이 아니다. 태스크별
결과와 실패 원인, 2주차 데이터 계획은 [Day 7 평가](docs/day07_evaluation.md)에 있다.
