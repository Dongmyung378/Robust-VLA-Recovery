# Failure-Aware VLA Project Charter

## 1. 문서 상태

- 프로젝트: Robust VLA Recovery
- Day 1: 2026-09-03
- 예정 종료일: 2026-10-21 (Day 49)
- 상태: 범위 동결
- 변경 원칙: 아래 범위는 주차별 Gate 실패 또는 명시된 축소 조건이 발생할 때만 변경한다. 변경 시 이유, 영향, 날짜를 이 문서의 결정 기록에 남긴다.

## 2. 연구 질문과 가설

### 핵심 연구 질문

> 실패 감지와 제한된 규칙 기반 복구가 교란 환경에서 VLA 로봇 조작의 최종 성공률을 높이는가?

### 사전 가설

- H1: `D Recover`는 같은 강건 학습 정책을 사용하는 `B Augmentation`보다 OOD Task Success Rate가 높다.
- H2: 시간축 감지기는 단일 프레임 감지기보다 failure onset 이전에 더 유용한 경고를 제공한다.
- H3: 복구는 일부 실패를 성공으로 바꾸지만, 오탐 복구와 harmful recovery라는 비용을 만든다.

가설이 지지되지 않아도 프로젝트 실패로 간주하지 않는다. 고정 프로토콜에서 실패 원인과 부작용을 재현 가능하게 설명하는 결과도 유효한 연구 결과다.

## 3. 고정 시스템 범위

- 시뮬레이터: LIBERO 계열 환경과 LIBERO-Plus 교란
- 기본 정책: SmolVLA 우선
- 자원 부족 시 정책: SmolVLA 설정 축소 후에도 불가능하면 ACT + frozen CLIP text encoder로 전환
- 감지 입력: 최근 K step의 RGB/VLA visual feature, proprioception, action 및 사용 가능한 confidence
- 실패 유형: `grasp_failed`, `object_dropped`, `wrong_object`, `stalled`, `collision`
- 복구 행동: `reobserve`, `backoff`, `regrasp`, `retry_subtask`, `abort`
- 복구 관리자: 규칙 기반 상태기계
- 상태: `NORMAL`, `SUSPECT`, `RECOVER`, `VERIFY`, `ABORT`
- 실험 범위: 시뮬레이션만 포함하며 실제 로봇 실험은 포함하지 않는다.

## 4. 고정 조작 태스크와 성공 조건

실제 LIBERO 환경 ID는 Day 3~4에 아래 의미와 가장 잘 맞는 태스크로 매핑한다. 환경 ID 매핑은 범위 변경이 아니며, 태스크 의미를 바꾸는 교체만 범위 변경으로 기록한다.

| ID | 태스크 | 성공 조건 | 실패 관찰의 핵심 |
|---|---|---|---|
| `pick_place` | 지정 객체를 목표 receptacle에 넣기 | 제한 시간 안에 목표 객체가 목표 영역에 있고 gripper에서 해제된 상태가 10 simulation step 연속 유지되며 환경의 native success predicate가 참이다. | grasp 실패, wrong object, drop |
| `stack` | 지정 객체를 목표 객체 위에 쌓기 | 상단 객체가 목표 객체에 안정적으로 지지되고 gripper에서 해제된 상태가 10 simulation step 연속 유지되며 native success predicate가 참이다. | 정렬 실패, drop, stalled |
| `open_drawer` | 지정 서랍 열기 | 제한 시간 안에 목표 drawer joint가 환경의 성공 임계값을 넘고 10 simulation step 동안 유지되며 native success predicate가 참이다. | wrong handle/object, stalled, collision |
| `shelf_place` | 지정 객체를 목표 선반 영역에 놓기 | 제한 시간 안에 목표 객체가 지정 shelf region 안에 있고 gripper에서 해제된 상태가 10 simulation step 연속 유지되며 native success predicate가 참이다. | grasp 실패, drop, collision |

공통 판정 규칙:

1. 공식 성능 지표는 환경의 native success predicate를 우선한다.
2. 10-step 안정성 검사는 일시적 접촉으로 인한 거짓 성공을 찾기 위한 진단값으로 함께 기록한다.
3. timeout horizon과 control frequency는 태스크 어댑터 구현 시 고정하고 모든 비교군에 동일하게 적용한다.
4. 성공 조건 변경은 기존 결과와의 비교를 깨므로 평가 시작 후에는 금지한다.

## 5. 비교군과 인과 질문

| 비교군 | 정책 학습 데이터 | Detector | Recovery | 답하려는 질문 |
|---|---|---|---|---|
| A `Base` | 정상 | 없음 | 없음 | 최소 기준선은 어느 정도인가? |
| B `Augmentation` | 정상 + 교란 | 없음 | 없음 | 강건 학습만으로 얼마나 개선되는가? |
| C `Detector` | 정상 + 교란 | 있음 | 기록만 | 감지 품질과 오탐 비용은 얼마인가? |
| D `Recover` | 정상 + 교란 | 있음 | 있음 | 감지 기반 폐루프 복구의 순기여는 얼마인가? |

핵심 비교는 `D - B`다. `B`와 `D`가 같은 정책 checkpoint와 평가 조건을 사용하게 해 복구 시스템의 기여를 분리한다. `A - B`는 augmentation 효과, `C`는 detector의 진단 성능을 설명하는 데 사용한다.

## 6. 평가 지표

### 1차 지표

- 정상 및 OOD Task Success Rate
- 핵심 효과량: `D Recover`와 `B Augmentation`의 OOD 성공률 절대 차이(percentage point)

### 2차 지표

- OOD Degradation
- Failure Macro-F1 및 AUROC
- failure onset 대비 detection lead time
- 정상 episode당 false trigger 수와 False Recovery Rate
- Recovery Rate 및 Harmful Recovery Rate
- Task Time, retry 횟수, policy/detector/manager별 latency와 총 Control Latency
- 태스크·교란·severity·seed별 원시 결과와 bootstrap 95% confidence interval

최종 성공 여부를 평균 하나로만 판단하지 않는다. 태스크별 분산, 교란별 성능, 정상 조건 손상 여부와 부정적 사례를 함께 보고한다.

## 7. 데이터 및 평가 분리 원칙

- 학습·검증·테스트는 episode가 아니라 seed와 scene 구성 단위로 분리한다.
- paraphrase와 perturbation template도 split 사이에서 중복되지 않게 관리한다.
- 정상/OOD와 wrong-instruction stress set을 분리해 보고한다.
- failure onset은 종료 결과만으로 정의하지 않고, 사건 기반 라벨과 감지 가능한 선행 구간을 보존한다.
- 평가 프로토콜이 동결된 Week 6부터는 모델, threshold 또는 recovery mapping을 변경하지 않는다.
- 실패하거나 손상된 episode의 재실행 규칙을 미리 정하고 모든 재실행을 로그에 남긴다.

## 8. 자원 예산

2026-09-03에 확인한 개발 장비:

| 자원 | 확인값 | 운영 결정 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM | 7B급 VLA 확장 제외. mixed precision, 작은 batch, gradient accumulation을 우선한다. |
| CPU | Intel Core i7-11800H, 16 logical processors | 환경 rollout과 전처리 worker 수는 발열·메모리를 보며 보수적으로 설정한다. |
| RAM | 약 63.9 GiB | feature cache와 데이터 검사는 가능하나 전체 RGB 데이터의 메모리 상주는 피한다. |
| C: 여유 공간 | 약 104.4 GiB | 코드, 환경, 작은 산출물만 둔다. |
| D: 여유 공간 | 약 1.49 TiB | Day 2에 데이터·checkpoint·대형 rollout의 외부 저장 위치 후보로 검증한다. |
| 일일 집중 작업 예산 | 4시간(임시 기본값) | 장시간 학습·rollout은 무인 실행하고 다음 작업일에 검증한다. 사용자 가용 시간이 확인되면 일정만 조정한다. |

학습 한 번은 24시간 안에 끝나는 설정만 본 실험 후보로 인정한다. 디스크 예상 사용량은 RGB 해상도, FPS, episode horizon을 확인한 Day 5에 다시 계산한다.

## 9. 범위 축소 규칙

범위가 일정 또는 자원을 초과하면 아래 순서를 지킨다. 앞 단계로 해결되면 뒤 단계는 적용하지 않는다.

1. 태스크를 4개에서 3개로 축소하고 `shelf_place`를 먼저 제외한다.
2. 교란 축을 5개에서 camera, object, language 3개로 축소한다.
3. 평가 seed를 조건당 5개에서 3개로 축소하되, 동일 seed를 모든 비교군에 사용한다.
4. SmolVLA의 image resolution, sequence length, batch size를 줄이고 gradient accumulation을 사용한다.
5. SmolVLA 본학습이 24시간 예산을 넘거나 6GB VRAM에서 안정화되지 않으면 ACT + frozen CLIP text encoder로 전환한다.
6. 다중 실패 유형 detector가 불안정하면 binary failure prediction을 먼저 완성한다.
7. 학습형 recovery로 확장하지 않고 규칙 기반 manager의 엄정한 평가를 유지한다.

어떤 축소에서도 `B Augmentation`과 `D Recover`의 공정한 비교, 정상/OOD 분리, false/harmful recovery 보고는 제거하지 않는다.

## 10. 비범위

- 실제 로봇 배포와 sim-to-real 성능 주장
- OpenVLA급 7B 모델의 full fine-tuning
- 학습형 또는 LLM 기반 recovery planner
- 네 비교군 밖의 대규모 모델 탐색
- 평가 결과를 본 뒤 seed, 성공 조건 또는 보고 지표를 유리하게 변경하는 행위
- 프로젝트 핵심 질문과 직접 관련 없는 UI 또는 서비스 배포

## 11. 주차별 Gate

| 주차 | 종료 시 반드시 남아야 하는 증거 |
|---|---|
| Week 1 | 네 태스크 실행 영상, baseline 결과, 손상 없이 재생되는 rollout 파일 |
| Week 2 | 동결 checkpoint, 학습 로그, 언어 조건 평가 |
| Week 3 | 교란 config, 자동 실패 라벨, 버전 데이터셋 통계 |
| Week 4 | detector checkpoint, Macro-F1/AUROC/calibration/lead-time/false-trigger 결과 |
| Week 5 | 폐루프 영상, recovery event log, Base 대 Recover 파일럿 비교 |
| Week 6 | 고정 eval manifest, 네 비교군 결과표, 95% 신뢰구간 |
| Week 7 | README, 기술 보고서, 데모 영상, 재현 절차, release tag |

## 12. Day 1 완료 점검

- [x] 핵심 연구 질문을 한 문장으로 고정했다.
- [x] 네 태스크와 태스크별 성공 조건을 정의했다.
- [x] GPU, VRAM, RAM, 저장공간과 임시 일일 작업 예산을 기록했다.
- [x] 비교군과 핵심 평가 지표를 고정했다.
- [x] 태스크, 교란, seed, 모델, detector, recovery의 축소 순서를 정했다.
- [x] 결과가 음성일 때도 보고한다는 원칙을 명시했다.

## 13. 결정 기록

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-09-03 | 네 번째 태스크로 insertion 대신 `shelf_place` 선택 | 6GB VRAM 장비와 7주 일정에서 정밀 접촉·물리 튜닝 위험을 낮추면서 공간적 배치 실패는 유지한다. |
| 2026-09-03 | 핵심 비교를 `D Recover - B Augmentation`으로 고정 | 같은 강건 정책을 사용해 detector+recovery의 순기여를 분리한다. |
| 2026-09-03 | 일일 집중 작업 예산을 임시 4시간으로 설정 | 사용자 가용 시간이 아직 확인되지 않았으므로 보수적인 기본값을 사용한다. |
| 2026-09-03 | 실행 기준을 WSL2/Linux, Python 3.12, LeRobot 0.6.1로 고정 | 최신 LeRobot의 LIBERO extra가 Linux 전용이며 SmolVLA와 함께 관리되는 통합 경로를 제공한다. |
| 2026-09-05 | Day 4의 네 태스크를 `libero_90`의 name으로 매핑하고 400 step / 20 Hz로 고정 | 실제 BDDL의 language와 goal을 대조했다. ID는 각각 46, 16, 7, 86이며 상세 계약과 실행 증거는 `docs/day04_evaluation.md`에 기록했다. |

## 14. 다음 작업

Day 2에는 저장소 골격(`src`, `configs`, `scripts`, `tests`, `docs`, `assets`, `outputs`)을 만들고, Python 버전과 의존성 관리 방식, 공통 seed/device/output 설정, import smoke test 및 config validation을 구현한다.
