# 연구 프로토콜

[English](research-protocol.md) | [프로젝트 README](../README.ko.md)

## 목표

이 프로젝트는 시간축 실패 감지와 제한된 규칙 기반 복구가 분포 변화 환경의 VLA 로봇
조작 성능을 높이는지 검증한다. 모든 비교에는 같은 태스크 정의, 성공 조건, rollout
예산과 보고 규칙을 사용한다.

## 가설

- 실패 감지 후 복구하는 강건 정책은 같은 정책을 복구 없이 사용할 때보다 OOD 조건의
  성능이 높을 것이다.
- 시간축 감지기는 단일 frame 감지기보다 실패 전에 더 유용한 경고를 제공할 것이다.
- 복구는 일부 실패를 성공으로 바꾸지만 오탐 개입이나 해로운 개입도 만들 수 있다.

고정된 프로토콜이 실패 양상과 개입 비용을 명확히 보여 준다면 부정적 결과도 유효하다.

## 고정 범위

- Simulator: LIBERO 태스크와 LIBERO-Plus 방식의 교란
- 우선 정책: SmolVLA
- 자원 부족 시 대안: 작은 SmolVLA 설정, 이후 frozen CLIP text encoder를 사용한 ACT
- 실패 유형: `grasp_failed`, `object_dropped`, `wrong_object`, `stalled`, `collision`
- 복구 행동: `reobserve`, `backoff`, `regrasp`, `retry_subtask`, `abort`
- 복구 상태: `NORMAL`, `SUSPECT`, `RECOVER`, `VERIFY`, `ABORT`
- 배포 범위: simulation only

## 태스크와 성공 조건

| 태스크 | 공식 목표 | 추가 진단 |
|---|---|---|
| `pick_place` | Alphabet soup가 basket 안에 있음 | 목표 물체가 10스텝 연속으로 놓인 상태 |
| `stack` | 앞쪽 검은 bowl이 가운데 bowl 위에 있음 | 목표 물체가 10스텝 연속으로 놓인 상태 |
| `open_drawer` | 위쪽 drawer가 LIBERO의 열림 기준을 통과함 | 공식 성공이 10스텝 동안 유지됨 |
| `shelf_place` | 가운데 book이 위쪽 shelf 영역에 있음 | 목표 물체가 10스텝 연속으로 놓인 상태 |

LIBERO의 기본 predicate가 공식 태스크 성공을 결정한다. 10스텝 신호는 일시적 접촉을
진단하며 benchmark predicate를 대체하지 않는다. 모든 태스크는 20 Hz에서 최대
400스텝을 실행한다.

## 비교군

| 그룹 | 정책 데이터 | 감지기 | 복구 | 질문 |
|---|---|---|---|---|
| A `Base` | 정상 | 없음 | 없음 | 정상 데이터 학습 정책의 하한은 얼마인가? |
| B `Augmentation` | 정상과 교란 | 없음 | 없음 | 강건 학습만으로 얼마나 개선되는가? |
| C `Detector` | 정상과 교란 | 기록만 | 없음 | 감지 trigger의 정확도와 비용은 얼마인가? |
| D `Recover` | 정상과 교란 | 사용 | 사용 | 복구 시스템의 순기여는 얼마인가? |

주 비교값은 D에서 B를 뺀 차이다. 두 그룹은 같은 정책 checkpoint와 평가 조건을
사용해 detector와 recovery system의 차이만 남겨야 한다.

## 평가 지표

주 결과는 정상 조건과 OOD 조건의 태스크 성공률, 그리고 D와 B 사이 OOD 성공률의 절대
차이다. 보조 지표는 다음과 같다.

- 정상 조건 대비 OOD 성능 저하
- 실패 유형별 Macro-F1과 AUROC
- 실패 시작 시점 대비 감지 선행 시간
- 정상 에피소드의 오탐 trigger와 오탐 복구
- 복구율과 해로운 복구율
- 태스크 시간, 재시도 횟수, component latency와 전체 제어 latency
- 태스크, 교란, 강도와 seed별 결과 및 bootstrap 95% confidence interval

## 데이터 분리

train, validation, test는 개별 frame이 아니라 source seed와 초기 장면 상태를 기준으로
group을 만든다. 의미가 같은 paraphrase와 교란 template도 데이터셋에 추가될 때 서로
다른 split으로 분리한다. 이 프로젝트가 감사한 공식 데모 파일에는 생성 seed가 없으므로
사용 가능한 누수 경계로 scene ID와 initial-state SHA-256을 사용한다. split 순서는
seed `378`로 결정한다.

일반 OOD 평가, 잘못된 instruction stress test와 학습 교란은 서로 분리한다. 실패하거나
손상된 에피소드는 조용히 다시 만들지 않고 제외 또는 재실행 이유를 기록한다.

## 자원 제약

개발 장비는 NVIDIA RTX 3060 Laptop GPU 6 GiB VRAM, RAM 약 64 GiB와 Intel Core
i7-11800H다. 전체 학습은 24시간 안에 끝나야 한다. RGB 데이터는 전체를 메모리에
올리지 않고 순차적으로 읽고 쓴다.

시간이나 계산 자원이 부족하면 다음 순서로 범위를 줄인다.

1. `shelf_place`를 제외하고 태스크 세 개를 유지한다.
2. 카메라, 물체와 언어 교란만 유지한다.
3. 조건별 평가 seed를 5개에서 3개로 줄이되 paired seed는 유지한다.
4. 이미지 해상도, sequence 길이와 batch size를 줄이고 gradient accumulation을 추가한다.
5. SmolVLA를 frozen CLIP text encoder를 사용하는 ACT로 바꾼다.
6. multi-class detection보다 binary failure prediction을 먼저 완성한다.
7. learned planner를 추가하지 않고 규칙 기반 recovery manager를 유지한다.

B와 D 비교, 정상과 OOD 보고, 오탐 복구와 해로운 복구 측정은 제거하지 않는다.

## 한계

이 프로토콜은 실제 로봇 배포, 7B VLA 전체 fine-tuning, learned recovery planner 또는
최종 test 결과를 확인한 뒤의 model selection을 다루지 않는다. 성공 조건, test seed와
보고 지표는 최종 평가 전에 고정한다.
