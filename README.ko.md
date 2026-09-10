# Robust VLA Recovery

[English README](README.md) | [연구 프로토콜](portfolio/research-protocol.ko.md) | [데이터 카드](portfolio/data-card.ko.md) | [학습 데이터 형식](portfolio/training-format.ko.md) | [실행 기록 형식](portfolio/rollout-format.ko.md)

Robust VLA Recovery는 시간축 실패 감지와 제한된 복구 행동이 분포 변화 환경에서
VLA 로봇 조작의 성공률을 높이는지 연구하는 프로젝트다. LIBERO 태스크에서 정책
추론, 실패 감지, 복구, OOD 평가를 하나의 재현 가능한 실험 규약으로 연결한다.

## 연구 질문

> 시간축 실패 감지와 제한된 규칙 기반 복구가 교란 환경의 최종 태스크 성공률을
> 높이는가?

정상 데이터 학습, 교란 데이터 증강, 감지 결과만 기록하는 조건, 감지 후 복구하는
조건을 비교한다. 태스크 성공률을 주 결과로 사용하고 오탐 복구, 해로운 복구,
감지 선행 시간과 제어 지연시간으로 개입 비용을 함께 측정한다.

## 시스템 구성

```text
언어 + RGB + 로봇 상태
          |
          v
      VLA 정책 ------> 행동 ------> LIBERO 환경
                                         |
                                         v
                                시간축 실패 감지기
                                         |
                                         v
                                   복구 관리자
                                         |
                                수정된 행동 또는 중단
```

감지기는 최근 시점의 시각 특성, 로봇 상태와 행동으로 구성된 인과적 시간창을
사용한다. 복구 관리자는 `reobserve`, `backoff`, `regrasp`, `retry_subtask`,
`abort`를 선택하는 규칙 기반 상태 기계다.

## 조작 태스크

| 키 | LIBERO 태스크 | 대표 실패 |
|---|---|---|
| `pick_place` | alphabet soup를 basket에 넣기 | 잡기 실패, 다른 객체 선택, 떨어뜨림 |
| `stack` | 앞쪽 검은 bowl을 가운데 bowl 위에 쌓기 | 정렬 실패, 떨어뜨림, 정지 |
| `open_drawer` | cabinet의 위쪽 drawer 열기 | 다른 손잡이 선택, 정지, 충돌 |
| `shelf_place` | 가운데 book을 cabinet shelf에 놓기 | 잡기 실패, 떨어뜨림, 충돌 |

모든 태스크는 `libero_90`에서 실행한다. 한 에피소드의 제한은 400 스텝이며 제어
주기는 20 Hz다. 공식 성공은 LIBERO의 기본 조건으로 판정하고, 10 스텝 안정성
신호는 별도 진단값으로 기록한다.

## 비교 실험

| 비교군 | 학습 데이터 | 감지기 | 복구 | 목적 |
|---|---|---|---|---|
| A `Base` | 정상 | 없음 | 없음 | 정상 정책의 하한 확인 |
| B `Augmentation` | 정상과 교란 | 없음 | 없음 | 강건 학습만의 효과 측정 |
| C `Detector` | 정상과 교란 | 기록만 | 없음 | 감지 품질과 오탐 측정 |
| D `Recover` | 정상과 교란 | 사용 | 사용 | 복구의 순기여 측정 |

핵심 비교는 같은 강건 정책을 사용하는 D와 B다. 성공 조건, 평가 지표와 범위 제약은
[연구 프로토콜](portfolio/research-protocol.md)에 정리했다.

## 현재 구현

- 재생 가능한 영상과 궤적 메타데이터를 남기는 결정적 LIBERO 태스크 어댑터
- 체크섬과 T/T+1 정렬 검사를 포함한 무손실 HDF5 실행 기록
- 제어 경로 확인에만 사용하는 결정적 경량 정책
- 네 태스크와 40개 에피소드로 구성된 정상 조건 기준선 평가
- 공식 데모 해시 고정과 누수 방지 분할을 포함한 데이터 감사기
- 에피소드와 언어 정렬을 명시한 결정적 정책 학습 포맷 변환기

현재 경량 정책은 학습되지 않았으며 VLA 성능을 나타내지 않는다. 공식 데모 변환을
마쳤으므로 다음 단계는 Day 10 단일 태스크 smoke fine-tuning이다.

## 공식 데모 감사

선택한 LIBERO 파일 네 개는 Hugging Face 리비전
`f13aa24a3da8c43c7225569f28c562979fa0e35a`로 고정했다. 현재 감사에서 데모
200개와 transition 26,145개가 모두 유효했고 제외 대상과 태스크 불균형은 없었다.
길이가 긴 에피소드 5개는 두 카메라의 시작·중간·종료 frame을 확인했으며 빈 화면이나
정지 화면은 발견되지 않았다.

| 태스크 | 데모 | Transition | 스텝 최소 / 중앙 / 최대 | 학습 / 검증 / 검사 |
|---|---:|---:|---:|---:|
| `pick_place` | 50 | 6,939 | 113 / 138 / 173 | 40 / 5 / 5 |
| `stack` | 50 | 6,415 | 103 / 126 / 191 | 40 / 5 / 5 |
| `open_drawer` | 50 | 4,736 | 67 / 89 / 177 | 40 / 5 / 5 |
| `shelf_place` | 50 | 8,055 | 132 / 160 / 211 | 40 / 5 / 5 |

공식 HDF5에는 생성 seed가 없다. 분할 seed `378`로 순서를 결정하고, 동일한 초기
상태 해시가 여러 split에 들어가지 않게 묶는다. 출처, 검사 규칙, 제외 기준과
라이선스 주의 사항은 [데이터 카드](portfolio/data-card.ko.md)에 있다.

## 정책 학습 데이터

Day 9에서 데모 200개와 transition 26,145개를 무손실 HDF5 학습 데이터 하나로
변환했다. 각 transition에는 원본 RGB 관측 두 개, 15차원 로봇 상태, 7차원 행동,
frame index, 정확한 20 Hz timestamp, frame과 에피소드 지시문을 연결하는 language
index가 들어 있다. 누적 offset으로 200개 에피소드 경계를 보존한다.

독립적으로 두 번 변환한 1,192,042,683 byte HDF5의 SHA-256은 모두
`f8ec588217d3c5a19f350394d01f428d9380efd1ae1e2e74a53ca2d3ff082da0`이었다. 모든
태스크와 split을 포함한 표본 10개에서 원본 pixel, state, action이 선언된 `float32`
변환 후 정확히 일치했다. 스키마와 검사 방법은
[학습 데이터 형식](portfolio/training-format.ko.md)에 정리했다.

## 빠른 시작

핵심 검사는 Python 3.12에서 실행하며 시뮬레이터를 설치하지 않는다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

LIBERO 실행은 WSL2의 Ubuntu 24.04 환경을 사용한다.

```bash
conda env create -f environment.yml
conda activate robust-vla-recovery
python -m pip install uv==0.12.9
uv pip sync requirements/simulation-linux.lock --torch-backend cu128 --no-build-isolation
uv pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## 주요 명령

| 명령 | 기능 |
|---|---|
| `rvla-run-libero-smoke configs/tasks/libero_smoke.toml` | 화면 없이 LIBERO 에피소드 하나 실행 |
| `rvla-run-tasks --task all --seed 378` | 네 태스크 어댑터 실행 |
| `rvla-rollouts collect --config configs/rollout.toml` | 무손실 실행 기록 수집 |
| `rvla-policy-rollout run --config configs/policy.toml` | 정책과 시뮬레이터의 제어 연결 검사 |
| `rvla-baseline-eval run --config configs/baseline_eval.toml` | 정상 조건 기준선 평가 |
| `rvla-demo-audit plan` | 고정된 공식 데이터 목록 확인 |
| `rvla-demo-audit run` | 데모 감사와 결정적 split 생성 |
| `rvla-demo-audit verify outputs/day08/<run>/audit.json` | 감사 결과와 원본 해시 재검증 |
| `rvla-convert-demos plan` | 고정된 정책 학습 변환 규약 확인 |
| `rvla-convert-demos run` | 감사된 데모 변환 |
| `rvla-convert-demos verify outputs/day09/<run>/conversion.json` | 원본, 스키마, 경계와 체크섬 재검증 |
| `rvla-convert-demos sample-check outputs/day09/<run>/conversion.json --count 10` | 원본과 변환본 표본 10개 대조 |

공식 HDF5 네 파일은 `data/libero_90/`에 둔다. `data/`와 `outputs/`는 모두 Git 추적
대상이 아니다.

## 저장소 구성

```text
configs/       실행, 태스크, 평가, 데이터 감사와 변환 설정
portfolio/     영어 기본 공개 문서와 같은 내용의 한국어 문서
src/           설치 가능한 Python 패키지
scripts/       시뮬레이터 계약 검사
tests/         단위 검사와 무결성 검사
data/          공식 데모, 로컬 전용
outputs/       영상, 실행 기록, 평가와 감사 결과, 로컬 전용
local_notes/   일차별 기록과 내부 결정, 로컬 전용
```

## 재현성과 한계

새 실행의 기본 seed는 `378`이다. 소스 리비전, 데이터 크기, 파일 해시, 태스크 정의와
split 개수는 버전 관리되는 설정에 고정한다. 대형 데이터와 실행 증거는 로컬에만
보관해 공개 저장소의 검토 범위를 코드와 재현 규약으로 제한한다.

현재 범위는 시뮬레이션뿐이다. 실제 로봇 전이를 주장하지 않으며, 현재 기준선 결과를
학습된 VLA 성능으로 해석해서는 안 된다. Day 9 tokenizer는 정렬과 vocabulary 출처를
검사하기 위한 것이며, 모델 전용 tokenizer 적용은 Day 10 범위다.
