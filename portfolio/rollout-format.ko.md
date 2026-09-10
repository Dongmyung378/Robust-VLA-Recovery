# 무손실 Rollout 형식

[English](rollout-format.md) | [프로젝트 README](../README.ko.md)

## Transition 정렬

두 개의 LeRobot 원본 `uint8` RGB stream을 resize, flip 또는 video compression 없이
저장한다. 두 화면을 나란히 배치한 H.264 파일은 확인용이며 학습 데이터로 사용하지
않는다.

action이 T개인 에피소드는 observation T+1개를 포함한다. observation 0은 reset과
10번의 안정화 스텝 이후 상태다. 20 Hz 정렬은 다음과 같다.

```text
observation[0], language[0] -- action[0] --> observation[1], reward[0], success[0]
       t = 0 ns                                  t = 50,000,000 ns
observation[1], language[1] -- action[1] --> observation[2], reward[1], success[1]
```

timestamp는 실제 시각이 아니라 결정적 simulation offset이다. action t는 observation과
language t를 사용하고, reward와 success t는 그 결과인 observation t+1을 설명한다.
마지막 observation 뒤에는 padding action을 추가하지 않는다.

## 디렉터리 규약

```text
outputs/day05/batch_<UTC timestamp>_<random suffix>/
  batch.json
  verification.json
  <task>_seed<S>_init<I>_lang<L>/
    episode.h5
    metadata.json
    preview.mp4
```

episode ID는 batch 안에서 고유하다. 새 수집은 새 디렉터리를 만들고 기존 에피소드를
덮어쓰지 않는다. `batch.json`에는 실행 전에 전체 task, seed, initial-state와
instruction matrix를 기록한다. 상태는 `running`에서 `complete` 또는 `failed`로
바뀐다. 오류는 보존하며 자동 재시도하지 않는다.

## HDF5 스키마

| Dataset | Shape | 타입과 의미 |
|---|---|---|
| `observations/image` | `(T+1, H, W, 3)` | 원본 agent-view `uint8` RGB |
| `observations/image2` | `(T+1, H, W, 3)` | 원본 wrist-view `uint8` RGB |
| `observations/proprio` | `(T+1, 34)` | `float64` robot state |
| `observations/language` | `(T+1,)` | UTF-8 task instruction |
| `observations/frame_index` | `(T+1,)` | 0부터 T까지의 정수 |
| `observations/timestamp_ns` | `(T+1,)` | frame index에 50,000,000을 곱한 값 |
| `transitions/action` | `(T, 7)` | `[-1, 1]` 범위의 유한한 `float32` 값 |
| `transitions/reward` | `(T,)` | 유한한 `float64` reward |
| `transitions/step_index` | `(T,)` | 0부터 T-1까지의 정수 |
| `transitions/terminated`, `truncated` | 각각 `(T,)` | Boolean 종료 신호 |
| `transitions/is_success`, `native_success` | 각각 `(T,)` | 누적 성공과 현재 공식 성공 |
| `transitions/stable_success`, `target_released` | 각각 `(T,)` | 안정성 진단 신호 |

34개 proprioception 값은 end-effector position, quaternion, rotation matrix, gripper
position과 velocity, 일곱 joint의 position과 velocity로 구성된다. 이 순서는 episode
metadata에도 기록한다.

## 출처 기록

metadata와 HDF5 attribute에는 episode ID, task name과 ID, LIBERO suite, seed, initial
state, instruction index, horizon, task configuration, BDDL과 catalog hash, dependency version,
policy, image orientation과 timestamp 규칙을 저장한다.

현재 수집 규약은 `{enabled: false, template_id: clean-v1}`로 명시한 정상 조건만 지원한다.
지원하지 않는 교란은 실행된 것처럼 기록하지 않고 거부한다. 이 rollout은 logging 경로를
검사하기 위한 것이며 expert demonstration이나 독립된 evaluation set이 아니다.

## 무결성과 중단 처리

writer는 매번 추가하기 전에 index, camera 존재 여부, 고정된 image shape, 유한한 robot
state, 유한한 reward, 범위 안의 action, language와 Boolean flag를 검사한다. transition
마다 flush하며 전체 에피소드를 메모리에 모으지 않는다.

작성 중인 파일명은 `episode.partial.h5`이고 metadata 상태는 `incomplete`다. 정상 종료는
end signal과 전체 스키마 검사를 통과해야 하며 이후 파일명을 `episode.h5`로 바꾼다.
예외나 끝나지 않은 context는 상태를 `aborted`로 바꾼다. process 또는 전원이 갑자기
종료되면 `incomplete` 파일이 남을 수 있고 verifier는 이를 거부한다.

검증 범위는 다음과 같다.

- JSON과 HDF5의 완료 marker와 출처 정보 일치
- 전체 파일 SHA-256과 수집 당시 RGB pixel hash
- dataset 길이, dtype, index와 timestamp
- 마지막 transition에만 존재하는 종료 신호
- `is_success`와 `native_success` 누적 OR의 일치
- 최종 결과와 stability 및 release 신호의 일치
- 모든 RGB frame의 압축 해제
- 설정된 task, seed, scene과 language matrix의 정확한 일치

hash는 우발적인 손상을 찾지만 데이터와 기록된 hash를 함께 의도적으로 바꾼 경우의
진위를 인증하지는 않는다.

## 압축과 저장 용량

RGB dataset은 gzip level 1, shuffle, Fletcher32와 frame 하나 단위 chunk를 사용한다.
저장 공간은 압축되지 않은 RGB 최솟값부터 계산한다.

```text
bytes_RGB = episodes x (horizon + 1) x cameras x height x width x 3
20 x 401 x 2 x 256 x 256 x 3 = 3,153,592,320 bytes, about 2.94 GiB
```

수집 전에는 이 RGB 예상치의 두 배와 다른 배열, HDF5 overhead, metadata를 위한 100 MB
여유 공간이 필요하다. 정적인 no-op 장면에서 얻은 압축률을 이후 교란 정책 rollout의
보장값으로 사용하지 않는다.

## 명령

```bash
rvla-rollouts collect --dry-run
rvla-rollouts collect --config configs/rollout.toml
rvla-rollouts verify outputs/day05/<batch>/batch.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

읽기와 검증에는 `requirements/data.txt`만 필요하다. MP4 export는 simulation 환경의
`imageio`와 `imageio-ffmpeg`를 추가로 사용한다.
