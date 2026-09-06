# Day 3 Evaluation - LIBERO Simulation and Episode Artifacts

## 결과

**판정: PASS (9.5/10)**

WSL2의 격리된 Python 3.12 환경에서 LIBERO 단일 태스크를 고정 시드와 고정
초기상태로 280 step 실행했다. EGL offscreen 렌더링으로 MP4를 생성했고, 관측·행동·
보상·종료 신호를 기록한 trajectory와 metadata를 저장했다. 독립 재생 스크립트가
영상 281 frame 전체 디코딩, trajectory 길이, 세 산출물의 SHA-256을 모두 검증했다.

## 평가표

| 평가 항목 | 배점 | 결과 | 근거 |
|---|---:|---:|---|
| Linux/WSL2 및 CUDA 환경 | 2.0 | 2.0 | Ubuntu 24.04 WSL2, Python 3.12.14, RTX 3060 Laptop GPU, CUDA 12.8을 확인했다. |
| 재현 가능한 simulation 설치 | 2.0 | 2.0 | LeRobot 0.6.1, hf-libero 0.1.4와 전체 전이 의존성 155개를 lock으로 고정했고 `pip check`를 통과했다. |
| 실제 LIBERO episode 실행 | 2.0 | 2.0 | `libero_spatial` task 0, seed 20260904, init state 0을 280 step 실행했다. |
| 인터페이스 기록 | 1.5 | 1.5 | 2개 RGB 관측, 34차원 raw robot state, 7차원 action, reward, terminated, truncated의 shape와 dtype을 metadata에 남겼다. |
| Headless 영상 생성 | 1.5 | 1.25 | EGL로 256×256 MP4 281 frame을 생성하고 육안 및 전체 디코딩을 확인했다. WSL의 `/dev/dri` 권한 경고로 EGL이 software fallback을 사용했다. |
| 재생 및 무결성 검증 | 1.0 | 0.75 | MP4 전체 디코딩, trace 길이, SHA-256 검증은 통과했다. 프레임별 simulator 재현성 비교는 후속 평가 범위다. |

## 기준 실행

- 실행 디렉터리: `outputs/day03/20260904T025952Z_libero_spatial_task00_seed20260904`
- 명령 태스크: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- 정책: fixed no-op `[0, 0, 0, 0, 0, 0, -1]`
- 실행 결과: 280 step, return 0.0, timeout truncation, success false
- 영상: H.264 MP4, 281 frame, 256×256×3, 20 fps
- wall-clock rollout: 32.750649초

`success=false`는 시뮬레이터 실패가 아니다. Day 3은 정책 성능 평가가 아니라 환경과
입출력 경로 검증 단계이며, 물체를 조작하지 않는 고정 no-op action이므로 예상된
결과다. 정책 success rate 기준선은 정책을 연결하는 다음 단계에서 측정한다.

## 기록된 인터페이스

| 항목 | shape | dtype |
|---|---|---|
| agent view | `(256, 256, 3)` | `uint8` |
| wrist view | `(256, 256, 3)` | `uint8` |
| flattened raw robot state trace | `(281, 34)` | `float64` |
| action per step | `(7,)` | `float32` |
| reward trace | `(280,)` | `float32` |
| terminated trace | `(280,)` | `bool` |
| truncated trace | `(280,)` | `bool` |

## 검증 증거

```text
Ran 9 tests in 0.075s
OK
No broken requirements found.

status: verified
steps: 280
frames: 281
frame_shape: [256, 256, 3]
video sha256: 5c9f439d8c21b275b6c2e68213e86481cd086d92d7d05068ba7bc82c2005ed1f
trajectory sha256: 639e1235686f8ef1edc2593ed6448145139fa2b25a8733b35b11448c2eebe6a2
first_frame sha256: 5320cb29c1f5869286129da02825c606e21f3cc25a1c936e2b596c1a53c39c91
```

## Day 3 Gate

- [x] WSL2에서 LIBERO와 LeRobot integration이 import된다.
- [x] CUDA가 전달되고 GPU가 식별된다.
- [x] 한 태스크를 고정 시드와 고정 초기상태로 끝까지 실행한다.
- [x] observation/action/reward/done 계열의 shape와 dtype을 기록한다.
- [x] GUI 없이 RGB frame과 MP4를 생성한다.
- [x] episode trajectory와 metadata를 저장한다.
- [x] 독립 재생 스크립트가 영상과 trajectory 무결성을 검증한다.
- [x] 실패한 임시 venv와 실패 run 산출물을 제거한다.

## 남은 위험과 다음 단계

1. WSL EGL은 정상 동작하지만 `/dev/dri/renderD128` 권한 문제로 software renderer에
   fallback했다. Day 3 기능에는 영향이 없지만 대규모 병렬 rollout 전에 GPU EGL
   설정을 최적화할 가치가 있다.
2. 고정 no-op episode는 정책 품질을 검증하지 않는다. 다음 단계에서 SmolVLA를
   연결하고 성공률과 latency를 별도로 측정해야 한다.
3. 현재 replay는 저장 영상 전체 디코딩과 데이터 무결성 검증이다. 동일 action을
   simulator에 다시 넣은 프레임별 결정성 검증은 후속 회귀 테스트로 확장할 수 있다.
