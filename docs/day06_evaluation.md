# Day 6 - Baseline policy 연결

## 판정과 범위

로드맵의 6일차 완료 기준을 충족했다. seed 378의 `pick_place` 환경에서 정책이
출력한 action으로 100 step을 연속 실행했고, 두 카메라 영상과 상태, action을
HDF5에 저장한 뒤 다시 검증했다.

설치된 LeRobot 0.6.1에는 SmolVLA 코드가 있지만 로컬 pretrained checkpoint는
없었다. 로드맵이 허용한 경량 baseline 경로를 사용해 입출력 연결부터 확인했다.
`lightweight_mlp_v1`은 seed로 고정한 미학습 정책이다. 이번 결과는 시스템 연결
검사이며 SmolVLA 또는 학습 정책의 성공률로 사용하지 않는다.

## 입출력 계약

- 입력은 agent와 wrist RGB, 34차원 proprioception, 선택한 instruction이다.
- RGB는 `uint8 / 255`로 `[0, 1]` 범위에 맞춘 뒤 카메라별 channel mean과
  standard deviation을 계산한다.
- 상태는 원소별 `tanh`를 적용한다. 언어는 재현 가능한 16-byte hash feature로
  바꾼다. 이 언어 feature는 데이터 흐름 검사용이며 의미 이해를 나타내지 않는다.
- 출력에는 `tanh`를 적용한다. arm 6축은 0.05, gripper는 1.0을 곱한 뒤
  `[-1, 1]`로 제한한다.
- 환경 제어 주기는 기존 task contract와 같은 20 Hz다. 초기 10회 호출은
  warmup으로 실행하고 latency 통계에서 제외한다.

설정 loader는 알 수 없는 field, 저장소 밖 경로, 잘못된 index, 범위를 벗어난
action scale, 100 step 미만의 실행을 거부한다.

## 실제 실행 결과

기준 metadata는
`outputs/day06/20260906T225619Z_pick_place_seed378_policy/metadata.json`이다.

| 항목 | 결과 |
|---|---:|
| task | `pick_place` |
| seed | 378 |
| action step | 100 |
| observation | 101 |
| RGB frame | 202 |
| 평균 inference latency | 5.463 ms |
| 중앙값 | 5.330 ms |
| p95 | 6.427 ms |
| 최댓값 | 7.781 ms |
| 50 ms 초과 | 0회 |
| success | false |
| HDF5 크기 | 19,211,402 bytes |
| HDF5 SHA-256 | `ec1433a3002ca48fb301907f035eaeefc32a4889a3c6ec909103bc1639069af5` |

100번째 step은 환경 실패가 아니라 정해 둔 `policy_rollout_limit`으로 끝냈다.
arm action은 관측에 따라 변했고 설정한 0.05 범위를 넘지 않았다. gripper 출력은
약 -0.983이었다.

## 실패 후보 검토

두 카메라 영상의 0, 25, 50, 75, 100 frame을 확인했다. arm은 움직였지만
alphabet soup를 잡지 못했고, 목표 물체는 테이블 위에 남아 있었다. 100번째
step의 native success도 false였다. metadata에는 `grasp_failed`와 `stalled`을
후보로 기록했다. 이는 수동 검토 메모이며 Day 20의 자동 label이나 정답 label이
아니다.

## 검증 결과와 다음 범위

- WSL에서 unit 및 integrity test 46개가 통과했다.
- HDF5 구조, timestamp, action 범위, provenance와 SHA-256을 확인했다.
- 두 카메라 202 frame을 모두 압축 해제해 수집 시 pixel hash와 비교했다.
- 수동 검토 내용을 추가한 뒤에도 episode 재검증이 통과했다.
- `pip check`에서 깨진 dependency가 없었다.

Day 7에서는 이 미학습 시스템 기준선을 네 태스크와 seed 10개로 확장해 성공률,
GPU 메모리와 episode latency를 측정한다. 학습된 정책과의 성능 비교는 Week 2
demonstration 학습 이후 같은 평가 계약으로 수행한다.
