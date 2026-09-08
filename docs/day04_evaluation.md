# Day 4 LIBERO 태스크 어댑터

## 판정과 범위

**판정: 통과.** 검증일: 2026-09-05.

네 태스크의 reset, 환경 성공, 시간 초과, 실패 처리, 언어 지시를 공통
인터페이스로 감쌌고 하나의 CLI에서 각각 또는 모두 실행할 수 있다. 실제
시뮬레이터에서 네 태스크를 총 1,600 step 실행하고 네 영상을 1,604 frame 모두
디코딩했다. 이 단계는 환경 어댑터를 검증하며 정책 품질은 평가하지 않는다. 이번
no-op 실행 네 개의 `success=false`를 정책 기준선 성공률로 보고하지 않는다.

## 태스크 매핑

모두 설치된 `hf-libero==0.1.4`의 `libero_90` 묶음을 사용한다. 숫자 ID를 코드에
직접 넣지 않고 아래 고유 이름으로 검색하며, 실행 metadata에 확인된 ID와
BDDL SHA-256을 기록한다.

| 프로젝트 key | 확인된 ID | 의미 | 환경 목표 |
|---|---:|---|---|
| pick_place | 46 | alphabet soup를 basket에 넣기 | `In alphabet_soup_1 basket_1_contain_region` |
| stack | 16 | 앞쪽 검은 그릇을 가운데 검은 그릇 위에 쌓기 | `On akita_black_bowl_1 akita_black_bowl_2` |
| open_drawer | 7 | cabinet의 위쪽 서랍 열기 | `Open wooden_cabinet_1_top_region` |
| shelf_place | 86 | 가운데 책을 cabinet shelf에 놓기 | `In black_book_1 wooden_two_layer_shelf_1_top_region` |

고유 이름은 [catalog](../configs/tasks/catalog.toml)에 보관한다. 태스크당 지시문
5개, 총 20개를 정의했으며 `--instruction-index 0..4`로 선택한다. 선택한 지시문은
`reset/step`의 `info.language`와 metadata의 `task.language`에 전달된다.
향후 학습·검사 언어 split은 별도로 정해야 하며, 이 20개를 독립 검사 자료로
간주해서는 안 된다.

쌓기 후보 중 `KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl`은
설치된 BDDL의 language가 “middle → front”인 반면 목표는 “middle → back”이었다.
지시문과 목표가 일치하는 앞쪽 → 가운데 태스크를 채택했다. 선택한 네 태스크는
시작할 때 catalog의 goal과 첫 지시문을 실제 BDDL과 대조하며 불일치 시 실패한다.

## 공통 인터페이스와 판정

`TaskAdapter(spec, backend, instruction_index=0, init_state_index=0)`:

- `reset(seed=S) -> (observation, info)`: 같은 seed와 명시된 초기상태를 다시 적용한다.
  LeRobot의 자동 초기상태 index 증가를 그대로 사용하지 않는다. 범위 밖 index는
  modulo로 바꾸지 않고 오류로 거부한다.
- `step(action) -> (observation, reward, terminated, truncated, info)`: 길이 7의
  유한한 action과 각 값 `[-1, 1]`을 요구한다. 잘못된 입력은 시뮬레이터를 진행시키지
  않고 `ValueError`를 반환한다.
- `render()`: agent 화면 RGB frame. 관측에는 agent와 wrist 이미지, 원본 로봇
  상태가 있다. 환경별 중복 코드는 `LiberoTaskBackend`에 모았다.
- `report_failure(kind)`: 외부 감지기나 작업자가 제공하는 이벤트의 기록과 종료 처리.
- `close()`: 자원을 해제하며 반복 호출 가능하다. close 후에는 새 adapter가 필요하다.

모든 태스크는 **400 action step / 20 Hz = 20초** 예산, reset 후 안정화 10 step,
안정성 진단 10 step을 사용한다. 안정화 단계는 episode action 예산에서 제외한다.

| 상황 | 반환/처리 |
|---|---|
| native predicate가 처음 참 | `is_success=true`를 유지하고 `first_success_step` 기록 |
| 현재 native predicate | `native_success`에 매 step 그대로 기록 |
| native success + target released가 10 step 연속 | `stable_success=true`, `terminated=true` |
| 일시 성공 뒤 predicate가 거짓 | 연속 안정성 횟수만 0으로 초기화하고 공식 성공 기록은 유지 |
| 400번째 step까지 안정성 확인 미완료 | `truncated=true`, reason=`timeout`; native 성공 기록은 별도로 유지 |
| 400번째 step에 안정성 확인 완료 | 안정성 종료 우선: `terminated=true`, `truncated=false` |
| backend 종료인데 환경 성공이 거짓 | 치명적 오류 `backend_terminated` |
| NaN/Inf 관측 또는 reward | 치명적 오류로 종료하고 시뮬레이터 예외는 이벤트 보존 후 전파 |
| reset 전 / 종료 후 step | `RuntimeError`, reset 필요 |

물체 배치의 해제 진단은 robosuite의 양쪽 fingerpad 접촉 기반 `_check_grasp`가
거짓인지를 사용한다. 위치·지지 여부는 native predicate를 따른다. 서랍은 별도
gripper release 없이 native joint predicate를 10 step 유지하는지 확인한다.
설치된 `WoodenCabinet.is_open`의 임계값은 `qpos < -0.14`이다. 위치 판정이나
grasp proxy가 완벽한 물리적 안정성을 보장한다고 주장하지 않는다.

진단을 위해 첫 성공 후에도 원래 예산 안에서 실행을 계속할 수 있다. 공식 task time은
`first_success_step / 20`이며 진단 완료 시간과 구분된다. `info.done`은 LIBERO 원본
신호이고 호출자는 episode 종료 판단에 adapter의 `terminated/truncated`를 사용한다.

## 실패 시 계속 진행할 조건

`grasp_failed`, `object_dropped`, `wrong_object`, `stalled`, `collision` 이벤트는
복구 가능 후보로 기록하고 예산 안에서 계속 진행한다. `unsafe_state`,
`unrecoverable_failure`, `operator_abort` 이벤트는 즉시 adapter를 비활성화하고
후속 step을 차단한다 (`last_info.aborted=true`, 종료 이유 보존).

이는 이벤트 **처리 규칙**이다. 다섯 실패를 자동 인식하는 라벨러와 감지기는
Day 20 이후 범위이며 이번 구현은 감지 성능을 주장하지 않는다. 예를 들어 단순
접촉과 위험한 충돌의 구분은 호출자가 별도 근거로 `unsafe_state`를 전달해야 한다.
CLI는 실행 오류를 요약 파일에 기록하고 나머지 태스크 결과도 보존한 뒤 0이 아닌
종료 code를 반환한다.

## 실행 증거

기준 실행과 같은 명령은 다음과 같다. WSL 시뮬레이션 환경의 저장소 루트에서 실행한다.

```bash
rvla-run-tasks --task all --seed 20260905
```

| 태스크 | 실행 step | 디코딩 frame | 종료 | 영상·trajectory 해시 검증 |
|---|---:|---:|---|---|
| pick_place | 400 | 401 | timeout | 통과 |
| stack | 400 | 401 | timeout | 통과 |
| open_drawer | 400 | 401 | timeout | 통과 |
| shelf_place | 400 | 401 | timeout | 통과 |

모두 초기상태 0, no-op `[0,0,0,0,0,0,-1]`, H.264 256×256 / 20 fps를 사용했다.
네 장면의 첫 frame을 직접 확인했다. trajectory에는 400개 action/reward/종료 신호와
401개의 34차원 robot state, native/stable/released 진단값이 있다.

로컬 검증 자료:

- [전체 실행 요약](../outputs/day04/summary_20260905T013041158730Z.json)
- [pick_place 영상](../outputs/day04/20260905T013049Z_libero_90_task46_seed20260905/episode.mp4)
- [stack 영상](../outputs/day04/20260905T013108Z_libero_90_task16_seed20260905/episode.mp4)
- [open_drawer 영상](../outputs/day04/20260905T013133Z_libero_90_task07_seed20260905/episode.mp4)
- [shelf_place 영상](../outputs/day04/20260905T013202Z_libero_90_task86_seed20260905/episode.mp4)
- [실제 어댑터 검증](../outputs/day04/adapter_checks_20260905T013348672341Z.json)

산출물은 `.gitignore`의 `outputs/**` 규칙으로 제외된다. 이 로컬 링크는 clone에
포함되지 않으며 위 명령을 다시 실행하면 새 timestamp의 결과가 생성된다.

## 추가 검증

핵심 단위 검사 22개가 통과했다. 시간 초과 직전과 직후, 성공 시점이 최대 step과
겹친 경우, 환경 성공과 안정성의 차이, 실패 이벤트, 예외, 잘못된 action과 언어
전달을 포함한다.

별도 `python scripts/check_task_adapters.py` 실제 시뮬레이터 검사도 네 태스크 모두 통과했다.

- 같은 seed/index로 step 후 reset을 반복했을 때 **전체 simulator state 최대 절대차 0**.
- 두 카메라 reset 이미지 모두 byte 단위 일치.
- 범위 밖 초기상태 index 거부.
- `stalled` 이벤트 후 진행, `operator_abort` 후 step 차단.
- 서랍 joint를 `-0.15`로 직접 설정한 **검사 전용 fixture**에서 실제 native predicate가
  참이 되었고, step 1 성공 기록 → step 10 안정성 종료를 확인했다.

fixture는 정책이 서랍을 연 결과가 아니며 성공률 계산에 포함하지 않는다. 나머지 세
태스크의 정책 성공 사례는 아직 검증하지 않았다. reset 동일성도 이 장비와 고정
버전에서 얻은 결과이며 다른 GPU나 버전 사이의 bit 단위 동일성을 보장하지 않는다.

## 완료 기준과 다음 단계

- [x] 네 태스크의 의미와 환경 ID·이름 연결 확인.
- [x] reset/success/timeout 공통 인터페이스 구현.
- [x] 태스크당 지시문 최소 5개 정의 및 전달 테스트.
- [x] 실패 종료와 계속 진행 조건 구분.
- [x] 하나의 CLI로 네 태스크 각각 실행 및 모든 영상 검증.

WSL에서 EGL 경고와 소프트웨어 렌더링 메시지는 여전히 출력된다. 기능 검증에는
영향이 없었지만 GPU 렌더링 가속을 확인한 결과는 아니다. 대형 시뮬레이션 의존성은
변경하지 않았다. 다음 Day 5에서는 시점별 RGB와 언어를 기록하는 정식 실행 로거,
누락 검사, episode 20개 검증과 디스크 예산을 구현한다.
