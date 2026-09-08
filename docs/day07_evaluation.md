# Day 7 - 정상 조건 baseline 평가

## 완료 판정

로드맵의 7일차 항목을 완료했다. 네 태스크에서 seed 378부터 387까지 각각 10회,
총 40개 episode를 실제 LIBERO 환경에서 실행했다. 모든 episode가 오류 없이 400
step을 마쳤고 결과 행렬, 집계값, 실패 영상 체크섬을 독립 검증했다.

평가 정책은 Day 6의 `lightweight_mlp_v1`이다. 두 RGB 카메라, 34차원 상태,
instruction을 받지만 학습되지 않은 결정적 시스템 기준선이다. 따라서 이번 성공률은
학습된 SmolVLA 성능이 아니라 연결이 끝난 시스템의 하한값으로 해석한다.

## 자동화 계약

`rvla-baseline-eval`은 다음 절차를 한 명령으로 수행한다.

1. 설정 파일에서 네 태스크와 서로 다른 seed 10개의 40회 행렬을 검사한다.
2. 각 episode를 실제 환경에서 실행하고 합성된 두 카메라 영상을 기록한다.
3. episode 시간, 정책 추론 latency, 50 ms 초과 횟수와 CUDA peak memory를 기록한다.
4. 태스크별 성공률과 전체 집계표를 만든다.
5. 실패 영상은 태스크별 첫 실패를 우선 선택하고 정확히 5개만 보존한다.
6. 사람이 지정한 frame과 실패 후보를 기록한 뒤 영상 SHA-256과 결과를 검증한다.

실행 중에는 episode가 끝날 때마다 `results.json`을 원자적으로 갱신한다. 실행 오류는
다음 episode를 막지 않고 결과에 남는다. 모든 실행이 끝나면 임시 영상과 선택되지
않은 영상 35개를 제거한다.

## 실제 결과

기준 결과는
`outputs/day07/eval_20260908T005301112825Z_74d9222c/results.json`이다. 전체 평가는
889.502초가 걸렸고 16,000 action step을 실행했다.

| task | 완료 | 성공 | 성공률 | 평균 episode 시간 | 평균 추론 latency | 50 ms 초과 |
|---|---:|---:|---:|---:|---:|---:|
| `pick_place` | 10/10 | 0 | 0% | 21.149 s | 5.212 ms | 0 |
| `stack` | 10/10 | 0 | 0% | 27.704 s | 5.170 ms | 0 |
| `open_drawer` | 10/10 | 0 | 0% | 21.857 s | 5.027 ms | 0 |
| `shelf_place` | 10/10 | 0 | 0% | 18.194 s | 5.169 ms | 0 |
| 전체 | 40/40 | 0 | 0% | 22.226 s | 5.144 ms | 0 |

장비에서 NVIDIA GeForce RTX 3060 Laptop GPU 6,441,926,656 bytes를 감지했다.
정책 계산은 NumPy CPU 경로이므로 PyTorch CUDA peak allocated와 reserved memory는
모두 0 bytes였다. CUDA 정책의 GPU 사용량으로 오해하지 않도록 이 값을 그대로
보존했다.

0% 성공률은 실행 실패가 아니다. 미학습 정책은 관측에 따라 bounded action을 냈지만
물체와 목표의 의미 있는 관계를 학습하지 않았고, 모든 episode가 horizon까지 진행된
뒤 성공 조건을 만족하지 못했다. Week 2에서는 demonstration으로 정책을 학습한 뒤
같은 평가 계약을 유지해 비교한다.

## 실패 영상 검토

0, 100, 200, 300, 400 frame을 확인했다. 아래 유형은 관찰을 정리한 후보이며 Day 20
자동 라벨의 정답으로 사용하지 않는다.

| episode | 실패 후보 | 관찰 |
|---|---|---|
| `pick_place_seed378_init0_lang0` | `grasp_failed`, `stalled` | gripper가 alphabet soup를 확보하지 못하고 목표 물체는 frame 400까지 테이블에 남았다. |
| `pick_place_seed379_init1_lang1` | `grasp_failed`, `stalled` | arm이 작업 공간을 지나갔지만 목표와 정렬하거나 접촉하지 못했다. |
| `stack_seed378_init0_lang0` | `grasp_failed`, `stalled` | 앞쪽 bowl을 들지 못했고 두 bowl은 끝까지 분리된 상태였다. |
| `open_drawer_seed378_init0_lang0` | `stalled` | top drawer handle을 잡아당기지 못해 drawer가 닫힌 상태로 남았다. |
| `shelf_place_seed378_init0_lang0` | `grasp_failed`, `stalled` | 가운데 book을 잡거나 shelf로 옮기지 못했다. |

선택된 MP4 5개만 기준 결과 폴더에 남겼다. 각 파일의 SHA-256은
`results.json`에 기록되어 있으며 verifier가 실제 파일과 다시 비교한다.

## Week 2 데이터 규모

유효 demonstration 목표는 태스크당 50개, 총 200개로 고정한다. 각 태스크는 train
40개, validation 5개, test 5개로 나눈다. 총 목표 길이는 80,000 transition이다.

Day 5에서 측정한 400-step HDF5 한 개의 평균 크기 64,119,799 bytes를 적용하면 원본
예상치는 12,823,959,800 bytes, 약 11.94 GiB다. 변환 metadata와 작업 여유 20%를
더한 저장공간 예산은 15,388,751,760 bytes, 약 14.33 GiB로 잡는다.

공식 source에 유효 episode가 부족하더라도 부족분을 합성, 복제하거나 split 사이에서
재사용하지 않는다. 제외 사유와 실제 개수를 기록하고 seed와 scene 단위 분리를
유지한다. 이 계획은 `results.json`에도 기계 판독 가능한 값으로 저장되며 verifier가
코드의 고정 계획과 일치하는지 확인한다.

## 검증 결과

- WSL2 실제 실행 40개가 모두 완료됐다.
- 결과 행렬은 네 태스크 곱하기 seed 10개와 정확히 일치한다.
- 실패 영상은 정확히 5개이며 모두 사람이 검토했다.
- 선택 영상의 SHA-256과 폴더 내 MP4 목록이 일치한다.
- 태스크별 집계값을 episode 원본에서 다시 계산해 같은 값임을 확인했다.
- Week 2 episode, transition, split, 저장공간 계획이 고정값과 일치한다.
