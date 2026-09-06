# Day 5 — Rollout logger

## 판정

**PASS — 로드맵의 5일차 완료 기준 충족.** 검증일: 2026-09-05.

2026-09-06 재개 시 완료된 수집·검증 결과를 보존하고, 남아 있던 `rvla-rollouts`
CLI의 editable 설치 반영을 마쳤다. 설치된 명령의 `collect --dry-run`과 `pip check`가
정상 종료했다. 에피소드 수집과 전체 재생 검증은 중복 수행하지 않았다.

네 태스크 × 다섯 seed의 **20개 episode**를 실제 LIBERO에서 생성하고 저장했다.
수집 프로세스의 개별 검증에 이어 별도 CLI 프로세스로 20개 전체를 다시 읽었다.
총 **8,000 actions / 8,020 observations / 16,040 RGB frames**가 길이·timestamp·
파일 SHA-256·저장 전후 픽셀 SHA-256 검사를 모두 통과했다.

## 구현 내용

1. `src/robust_vla_recovery/data/rollout.py`: 두 카메라 원본 RGB, 34차원
   proprioception, 7차원 action, 시점별 language, reward, success와 종료 신호를
   streaming HDF5로 기록한다. 원본 RGB는 무손실 gzip level 1로 보존한다.
2. `src/robust_vla_recovery/data/collection.py`: config 기반 수집, disk budget 사전
   검사, batch manifest, 개별/전체 검증, 파생 MP4 export를 제공한다.
3. `configs/rollout.toml`: 네 태스크, seed 20260906…20260910, 초기상태 index 0…4,
   지시문 index 0…4, 256×256 RGB, 명시적인 clean-v1 조건을 고정한다.
4. `scripts/run_rollouts.py`: collect / verify / replay 공통 CLI.
5. `tests/test_rollout.py`: 프레임 누락, step 중복, 시간축 불일치, byte 및 pixel 손상,
   NaN, 저장 오류, metadata 변조, 미완료 episode, 누락된 batch 항목을 검사한다.
6. CI에 GPU 없이 numpy/h5py만 설치하는 data-integrity job을 추가했다. 원격 CI는
   실행하지 않았으며 로컬 동등 테스트 결과를 아래에 기록했다.

저장 형식과 시간 정렬 계약은 [rollout_format.md](rollout_format.md)에 정리했다.
관측 T+1개와 action T개를 보존하며 마지막 관측에 가짜 action을 추가하지 않는다.
시뮬레이션 시간은 reset 및 settle 뒤 0부터 시작하고 20 Hz에서 정확히
50,000,000 ns 간격이다. 정책의 실제 처리 지연 시간을 뜻하지 않는다.

## 실제 수행 결과

기준 배치: `outputs/day05/batch_20260905T142442Z_b2caab45`

| 태스크 | episode | action | 원본 RGB frame (2 cameras) | 평균 HDF5/episode |
|---|---:|---:|---:|---:|
| pick_place | 5 | 2,000 | 4,010 | 71.31 MiB |
| stack | 5 | 2,000 | 4,010 | 59.02 MiB |
| open_drawer | 5 | 2,000 | 4,010 | 54.85 MiB |
| shelf_place | 5 | 2,000 | 4,010 | 59.41 MiB |
| 합계 | **20** | **8,000** | **16,040** | **61.15 MiB** |

20개 모두 400-step horizon에서 timeout으로 끝났다. no-op
`[0, 0, 0, 0, 0, 0, -1]` 정책이므로 `success=false`는 예상된 결과다.
이는 정상 저장/재생 검증용 데이터이며 정책의 조작 성공률이나 expert demonstration
품질을 평가한 결과가 아니다. 성공 표시의 저장/재생 경로는 별도의 합성 unit test로
검증했다. Day 4에서 확인한 실제 서랍 predicate 검사는 그대로 유지된다.

태스크별 5개 episode의 wall-clock 측정 합계는 약 **620.94초**였다. 환경 초기화,
시뮬레이션, 쓰기 및 개별 읽기 검증을 포함한다. 마지막 별도 전체 재검증은 이 합계에
포함되지 않으며 policy inference latency 수치로 해석할 수 없다.

## 용량과 압축 결정

| 항목 | 값 |
|---|---:|
| 원본 RGB payload | 3,153,592,320 bytes = 2.9370 GiB |
| HDF5 20개 합계 (기타 배열·HDF5 overhead 포함) | 1,282,395,980 bytes = 1.1943 GiB |
| 원본 RGB 대비 실제 파일 크기 비율 | 약 2.46 : 1 |
| 평균 HDF5 episode | 약 61.15 MiB |
| 시작 시 확인한 free space | 115,128,172,544 bytes |
| 사전 요구 free space | 6,407,184,640 bytes (약 5.97 GiB) |

압축은 **gzip level 1 + 한 프레임 단위 chunk + Fletcher32**로 결정했다. H.264보다
큰 파일을 허용하는 대신 학습·감지 분석용 원본 픽셀을 보존하고 프레임별 접근을
지원한다. 기록이 길어져도 RGB 전체를 RAM에 누적하지 않는다.

동일 길이 1,000 episode는 이번 측정 평균 기준 약 **59.72 GiB**의 HDF5가 된다.
다만 미래 교란·noise·다른 장면의 압축률은 달라질 수 있으므로 보장값이 아니다.
비압축 RGB만으로는 같은 1,000개에 약 **146.85 GiB**가 필요하다. 대규모 수집은
예산을 다시 계산하고 프로젝트 계약서의 D: 대형 저장소 후보를 확인해야 한다.
현재 20개 검증 자료는 workspace의 ignored outputs에 보존했다.

## 재검증 증거

```text
WSL full tests:
Ran 37 tests in 0.579s
OK                          # skip 없음

Standalone batch verification:
status: verified
episodes: 20
steps: 8000
rgb_frames: 16040
raw_rgb_bytes: 3153592320
stored_bytes: 1282395980

Derived side-by-side preview:
codec_name: h264
width: 512
height: 256
r_frame_rate: 20/1
nb_read_frames: 401
```

Windows `python -S`로 site-packages를 제외한 검사에서는 core 24개가 통과하고
데이터 의존성이 필요한 13개는 명시적으로 skip됐다. dry-run도 같은 환경에서
실행돼 무거운 simulation import 없이 계획을 확인할 수 있음을 검증했다.

재생 영상의 첫 frame을 직접 확인해 agent/wrist 화면을 대조했다. 영상은 저장된
HDF5에서 만들어졌으며 simulator를 재실행하거나 그림을 합성한 것이 아니다.

## 완료 기준

- [x] RGB, proprioception, action, language, reward, success를 시점별 저장.
- [x] episode ID, task ID/name, seed, initial state, instruction index,
  perturbation config 및 버전/hash를 metadata에 보존.
- [x] 프레임 누락, 길이 불일치, timestamp/index 오류 검출.
- [x] 실패/중단 기록을 정상 완료와 구분하고 자동 덮어쓰기 방지.
- [x] 압축 방식 결정과 디스크 예상/실측 용량 산정.
- [x] **20개 실제 episode가 손상 없이 저장·재생됨.**

## 결과 파일과 재현

- [수집 batch manifest](../outputs/day05/batch_20260905T142442Z_b2caab45/batch.json)
- [독립 전체 재검증 결과](../outputs/day05/batch_20260905T142442Z_b2caab45/verification.json)
- [두 카메라 재생 영상](../outputs/day05/batch_20260905T142442Z_b2caab45/pick_place_seed20260906_init0_lang0/preview.mp4)

```bash
# WSL simulation environment, repository root
python scripts/run_rollouts.py collect --config configs/rollout.toml
python scripts/run_rollouts.py verify outputs/day05/<new-batch>/batch.json
```

outputs는 Git에서 제외되므로 clone에 위 로컬 evidence가 포함되지 않는다. 새 수집은
고유 batch 디렉터리를 만들어 기존 검증 자료를 덮어쓰지 않는다.

## 정리와 남은 범위

### 2026-09-06 검토 후 보완

- 환경 `close()` 예외는 별도 `cleanup_error`로 기록한다. 기존 수집 오류 또는
  검증된 파일 정보는 보존하고, `batch.json` 저장 후 다음 에피소드를 진행한다.
  종료 오류가 있는 배치는 최종 `failed`로 처리하며 수집 명령은 실패를 반환한다.
- 배치 재검증은 실제 HDF5와 metadata 간 일치뿐 아니라 최초 수집 시
  `batch.json`에 기록한 SHA-256과의 일치도 요구한다. 원래 해시의 누락·형식 오류나
  불일치는 거부한다. 배치 manifest 자체까지 변경하는 공격에 대한 인증은 아니다.
- 회귀 테스트 5개 추가: 원래 해시 일치, 누락/잘못된 해시, HDF5와 metadata 동시 변경,
  정상 수집 후 종료 오류, 수집 오류와 종료 오류의 동시 발생.
  WSL 전체 **42개 테스트 통과, skip 0개**.
- 강화된 검증기로 기존 배치 **20개 / 8,000 steps / RGB 16,040 frames** 재검증 통과.
  원본 데이터 재수집이나 커밋 없이 검증 보고서만 갱신했다.

`.gitignore`의 기존 `data/` 규칙이 source package인 `src/robust_vla_recovery/data`도
숨기고 있어 `/data/`로 한정했다. 소스 로거는 버전 관리 대상이며 대형 루트 데이터는
계속 제외된다. 테스트 임시 파일은 자동 정리됐고 이번 배치의 partial/aborted 기록은
없다. 기존 Day 3/4 결과도 보존했다. 커밋은 생성하지 않았다.

EGL의 기존 fallback 경고는 남아 있으며 이번 작업은 GPU rendering 가속을 주장하지
않는다. 학습된 정책의 연결과 scaling/normalization/100-step inference 검증은 다음
**Day 6** 범위다. 이번 no-op 데이터는 해당 정책의 학습 성공을 보장하지 않는다.
