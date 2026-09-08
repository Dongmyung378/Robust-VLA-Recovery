# 무손실 실행 기록 저장 형식 v1

## 목적과 시간 정렬

Day 5는 HDF5를 학습과 문제 분석에 사용하는 원본 기록으로 삼는다. 두 카메라의
LeRobot `uint8` RGB를 크기 변경, 반전, 영상 압축 없이 무손실로 저장한다. 두 화면을
나란히 붙인 H.264 MP4는 사람이 확인하기 위한 파생 파일이며 원본 데이터가 아니다.

action이 T개인 episode에는 관측이 T+1개 있다. 관측 0은 reset과 안정화 10 step이
끝난 직후 상태다. 제어 주기가 20 Hz일 때 시간 관계는 다음과 같다.

```text
observation[0], language[0] -- action[0] --> observation[1], reward[0], success[0]
       t = 0 ns                                  t = 50,000,000 ns
observation[1], language[1] -- action[1] --> observation[2], reward[1], success[1]
```

timestamp는 실제 경과 시간이 아닌 시뮬레이션 시작 기준 나노초다. 안정화가 끝난
시점부터 0으로 시작한다. action t는 observation과 language t를 사용하고,
reward와 success t는 그 결과인 observation t+1을 설명한다. 지시문이 변하지 않아도
모든 관측 시점에 language를 저장한다. 마지막 관측 뒤에는 action이 없으며 채우기용
가짜 action을 만들지 않는다.

## 디렉터리 구성

```text
outputs/day05/batch_<UTC timestamp>_<random suffix>/
  batch.json
  verification.json                    # 별도 일괄 재생 검증 결과
  <task>_seed<S>_init<I>_lang<L>/
    episode.h5                         # 종료 신호 확인 뒤 공개되는 원본
    metadata.json                      # 완료 표시와 체크섬
    preview.mp4                        # 선택 사항인 확인용 영상
```

episode ID는 한 batch 안에서 유일하다. 전체 식별자는 batch 디렉터리와 episode ID의
조합이다. 새 수집은 기존 episode를 덮어쓰지 않고 새 batch 디렉터리를 만든다.
`batch.json`에는 수집 전에 태스크, seed, 초기 상태, 지시문 조합 전체를 기록한다.
모든 시도가 끝날 때까지 상태는 `running`이며 이후 `complete` 또는 `failed`가 된다.
오류는 자동 재시도하지 않고 그대로 남긴다.

## HDF5 데이터셋

| 데이터셋 | 크기 | 자료형과 내용 |
|---|---|---|
| `observations/image` | `(T+1, H, W, 3)` | `uint8` agent camera RGB 원본 |
| `observations/image2` | `(T+1, H, W, 3)` | `uint8` wrist camera RGB 원본 |
| `observations/proprio` | `(T+1, 34)` | `float64` |
| `observations/language` | `(T+1,)` | UTF-8 문자열 |
| `observations/frame_index` | `(T+1,)` | `int64`, 정확히 0부터 T까지 |
| `observations/timestamp_ns` | `(T+1,)` | `int64`, frame index × 50,000,000 |
| `transitions/action` | `(T, 7)` | `float32`, 유한값이며 `[-1, 1]` 범위 |
| `transitions/reward` | `(T,)` | 유한한 `float64` |
| `transitions/step_index` | `(T,)` | `int64`, 정확히 0부터 T-1까지 |
| `transitions/terminated`, `truncated` | 각각 `(T,)` | `bool` |
| `transitions/is_success`, `native_success` | 각각 `(T,)` | `bool`, 누적 성공과 현재 환경 성공 |
| `transitions/stable_success`, `target_released` | 각각 `(T,)` | `bool`, Day 4 진단값 |

34차원 로봇 상태의 순서는 end-effector 위치 3개, quaternion 4개, 방향 행렬 9개,
gripper 위치 2개, gripper 속도 2개, 관절 위치 7개, 관절 속도 7개다. 이 순서는
metadata에도 저장한다.

## 출처와 실행 조건

metadata와 HDF5 속성에는 같은 실행 정보를 넣는다. episode ID, 프로젝트 태스크 key,
LIBERO 태스크 이름·ID·suite, seed, 초기 상태와 지시문 index, episode 최대 길이,
태스크 설정, BDDL과 catalog 해시, 패키지 버전, 정책 정보, 원본 이미지 방향,
timestamp 규칙을 포함한다.

Day 5는 `perturbation = {enabled: false, template_id: clean-v1}`을 명시한다. 지원하지
않는 교란은 적용된 것처럼 기록하지 않고 거부한다. 교란 생성은 Week 3 범위다.
현재 자료는 no-op action으로 기록 경로를 확인한 데이터이며 expert demonstration이나
독립 평가 split이 아니다. 언어 조건 정책이 없으므로 지시문 5개를 썼다는 사실만으로
언어 이해 성능을 주장할 수 없다.

## 무결성과 중단 처리

로거는 디스크에 쓰기 전에 요청한 step index, 두 카메라, 이미지 크기 유지,
로봇 상태·action·reward의 유한값, action 범위, language와 boolean 값을 검사한다.
그 뒤 transition 하나와 결과 관측 하나를 기록하고 HDF5에 즉시 반영한다. RGB frame을
메모리에 계속 쌓지 않으므로 기록 길이가 늘어도 메모리 사용량이 함께 증가하지 않는다.

기록 중인 파일 이름은 `episode.partial.h5`이고 metadata 상태는 `incomplete`다.
예외가 발생하거나 `finish()` 없이 writer를 닫으면 부분 자료를 보존하고 상태를
`aborted`로 바꾼다. 완료하려면 종료 신호가 있어야 하며, T/T+1 구조 전체를 검사한
뒤에만 `episode.h5`로 이름을 바꾼다. metadata는 임시 JSON을 거쳐 교체하므로 JSON
일부만 기록된 상태가 완료로 인정되지 않는다. 프로세스나 전원이 갑자기 끊기면
`incomplete` 자료가 남을 수 있고 검증기는 이를 거부한다. 이 동작은 오류 발견과
근거 보존을 위한 것이며 자동 복구나 전원 장애 내구성을 보장하지 않는다.

검증기는 다음 항목을 확인한다.

- 완료 표시와 HDF5·metadata의 실행 정보 일치
- HDF5 전체 파일의 SHA-256
- 모든 데이터셋의 길이, 자료형, index와 timestamp
- 마지막 transition에만 있는 종료 표시와 가짜 종료 행 부재
- 누적 성공값과 `native_success` 누적 OR의 일치
- 최종 결과와 안정 성공·물체 해제 진단의 기본 일관성
- 두 카메라의 모든 frame 압축 해제와 수집 당시 원본 픽셀 SHA-256 일치
- 설정한 태스크·seed·scene·language 조합과 batch의 정확한 일치

해시는 우발적인 손상을 찾지만 산출물과 해시를 함께 고친 고의 변조를 인증하지는
않는다. index와 개수 검사는 기록 누락을 찾지만 시뮬레이터가 정상 index에 오래된
이미지를 반환했는지까지 판단할 수는 없다.

## 압축과 저장공간 계획

RGB는 HDF5 gzip level 1, shuffle, Fletcher32와 frame 하나 크기의 chunk를 사용한다.
H.264보다 파일이 크지만 픽셀을 잃지 않고 frame별 임의 접근과 일정한 메모리 사용량을
유지한다. 압축률을 추측하지 않고 frame 수와 RGB 크기로 비압축 하한을 계산한다.

```text
bytes_RGB = episodes × (horizon + 1) × cameras × height × width × 3
20 × 401 × 2 × 256 × 256 × 3 = 3,153,592,320 bytes ≈ 2.94 GiB
```

수집기는 시작 전에 이 RGB 용량의 두 배와 여유 공간 100 MB를 요구한다. 다른 배열,
HDF5 구조와 metadata 용량은 별도다. 장면과 noise에 따라 압축률이 달라지므로 no-op
episode의 실측 크기를 미래 교란 정책 실행의 보장값으로 사용하지 않는다.

## 실행 명령

시뮬레이션 환경의 저장소 루트에서 실행한다.

```bash
rvla-rollouts collect --dry-run
rvla-rollouts collect --config configs/rollout.toml
rvla-rollouts verify outputs/day05/<batch>/batch.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

읽기와 재생 검증에는 `requirements/data.txt`만 필요하다. MP4를 만들 때는 시뮬레이션
환경의 imageio와 imageio-ffmpeg도 필요하다. `--dry-run`과 핵심 모듈 가져오기는
데이터·시뮬레이션 의존성 없이 동작한다. GitHub CI의 데이터 전용 작업은 손상 검사가
GPU 설치에 우연히 의존하지 않는지 확인한다.
