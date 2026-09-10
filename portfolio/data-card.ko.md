# LIBERO 데모 데이터 카드

[English](data-card.md) | [프로젝트 README](../README.ko.md)

## 범위와 출처

이 문서는 Robust VLA Recovery의 정책 학습에 사용할 공식 LIBERO 데모 파일 네 개를
다룬다. 2026년 9월 9일에 `yifengzhu-hf/LIBERO-datasets` 리비전
`f13aa24a3da8c43c7225569f28c562979fa0e35a`를 감사했다. 원본 파일명, byte 크기와
SHA-256은 `configs/data_audit.toml`에 고정했다.

공식 출처:

- [공식 LIBERO 저장소](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [고정된 LIBERO 데이터 리비전](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90)

## 감사 대상

| 태스크 | 원본 장면 | 파일 크기 | 데모 | Transition | 스텝 최소 / 중앙 / 최대 |
|---|---|---:|---:|---:|---:|
| `pick_place` | `LIVING_ROOM_SCENE1` | 692,219,541 B | 50 | 6,939 | 113 / 138 / 173 |
| `stack` | `KITCHEN_SCENE2` | 641,644,880 B | 50 | 6,415 | 103 / 126 / 191 |
| `open_drawer` | `KITCHEN_SCENE1` | 472,881,442 B | 50 | 4,736 | 67 / 89 / 177 |
| `shelf_place` | `STUDY_SCENE4` | 800,762,622 B | 50 | 8,055 | 132 / 160 / 211 |
| 합계 | 장면 4개 | 2,607,508,485 B | 200 | 26,145 | 67 / 태스크별 상이 / 211 |

태스크마다 데모 50개로 수가 같다. 모든 유효 에피소드에는 `(T, 7)` action과
`(T, 128, 128, 3)` shape의 `uint8` RGB 배열 두 개가 있다. 필수 simulator state,
robot state, reward, terminal, end-effector, gripper와 joint 배열의 첫 번째 길이는 같다.

## 품질 감사

에피소드 200개가 구조적 제외 규칙을 모두 통과했다. 초기 상태 해시 중복과 태스크 수
불균형은 없었다. 다음 에피소드 다섯 개는 태스크별 강건 길이 기준을 넘었다. 각
에피소드의 시작, 중간, 마지막 시점에서 두 카메라 frame을 확인했고, 태스크 진행이
보였으며 빈 화면이나 정지 화면은 없었다.

| 태스크 | 에피소드 | 스텝 | 결정 |
|---|---|---:|---|
| `stack` | `demo_42` | 191 | 구조가 유효하므로 유지 |
| `open_drawer` | `demo_0` | 177 | 구조가 유효하므로 유지 |
| `open_drawer` | `demo_6` | 149 | 구조가 유효하므로 유지 |
| `open_drawer` | `demo_16` | 150 | 구조가 유효하므로 유지 |
| `open_drawer` | `demo_40` | 141 | 구조가 유효하므로 유지 |

길이 경고만으로 궤적 오류를 판단할 수 없다. 표본 검토 결과는 해당 에피소드 유지를
뒷받침하지만, 포맷 변환 단계의 frame 단위 의미 검사를 대체하지 않는다.

## 분할 규칙

태스크마다 train 40개, validation 5개, test 5개를 사용하며 전체 개수는 각각 160개,
20개, 20개다. 순서 해시에는 seed `378`을 사용한다. 원본 파일에 생성 seed가 없으므로
현재 fallback group은 파일의 scene ID와 에피소드 초기 simulator state의 SHA-256을
결합한다. group 전체는 하나의 split에만 배정한다.

manifest에는 없는 source seed를 `null`로 기록하며 에피소드 번호로 값을 만들지 않는다.
향후 원본 리비전이 seed를 제공하면 auditor가 scene key와 해당 seed를 함께 사용한다.

## 제외와 경고 규칙

다음 조건 중 하나라도 해당하면 에피소드를 제외한다.

- 필수 action, image, state, reward 또는 terminal 배열이 없거나 읽을 수 없다.
- 배열이 비어 있거나 유한하지 않은 값을 포함하거나 transition 길이가 서로 다르다.
- action이 7차원이 아니거나 `[-1, 1]` 범위를 벗어난 값을 포함한다.
- RGB stream이 `uint8` 타입과 `(T, 128, 128, 3)` shape을 만족하지 않는다.
- `num_samples`와 action 길이가 다르거나 초기 simulator state가 없다.

원본 파일의 byte 크기나 SHA-256이 고정된 리비전과 다르거나 BDDL identity가 일치하지
않으면 파일 전체를 거부한다. 선언된 데모 개수 또는 transition 합계가 HDF5 내용과
달라도 같은 방식으로 처리한다.

길이 이상치, 마지막 `done` 값의 false 또는 초기 상태 중복은 자동 제외가 아니라
경고다. 엄격한 자동 규칙이 유효한 사람의 데모를 제거할 수 있으므로 이 경우는 직접
검토한다.

## 저장과 라이선스

로컬 원본 파일 네 개는 약 2.43 GiB이며 Git에서 제외한다. 감사 보고서와 split
manifest도 로컬 산출물이다. 공식 LIBERO 저장소는 데모 데이터셋을 CC BY 4.0으로
설명하지만, Hugging Face 데이터 카드의 현재 metadata는 Apache 2.0으로 표시한다.
데모 파일을 재배포하기 전에 공식 조건을 다시 확인해야 한다.

## 알려진 한계

- 감사는 출처와 데이터 구조를 검증하며 모든 action의 최적성을 판정하지 않는다.
- 길이 이상치 다섯 개는 전체 frame이 아니라 세 시점에서 확인했다.
- 이 원본 리비전에서는 source seed 출처를 확인할 수 없다.
- 네 태스크가 서로 다른 장면 네 개를 사용하므로 일반적인 장면 다양성 benchmark는 아니다.
