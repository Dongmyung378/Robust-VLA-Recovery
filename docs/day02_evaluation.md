# Day 2 Evaluation - Repository and Reproducibility Skeleton

## 결과

**판정: PASS (9.5/10)**

Day 2의 최소 완료 기준인 새 Python 환경에서의 editable install, config
validation, import smoke test를 통과했다. 대형 simulation dependency의 실제
설치와 LIBERO 실행은 로드맵상 Day 3 범위로 남긴다.

## 평가표

| 평가 항목 | 배점 | 결과 | 근거 |
|---|---:|---:|---|
| 권장 저장소 구조 | 2.0 | 2.0 | `src`, `configs`, `scripts`, `tests`, `docs`, `assets`, `outputs`가 생성되고 역할이 README에 기록됐다. |
| Python 및 핵심 의존성 고정 | 2.0 | 1.5 | Python 3.12, LeRobot 0.6.1, hf-libero 0.1.4, torch 2.11.0, torchvision 0.26.0, ffmpeg 7.1.1을 고정했다. Linux에서 전체 resolver가 만든 전이 의존성 lock은 Day 3 설치 후 남겨야 한다. |
| 공통 실행 설정과 검증 | 2.0 | 2.0 | seed, device, output directory, deterministic flag, 태스크 목록을 TOML에 통합하고 strict validation을 구현했다. |
| 테스트와 최소 CI | 2.0 | 2.0 | import smoke, 정상 config, unknown key, 경로 이탈, 중복 태스크 테스트와 GitHub Actions workflow를 추가했다. |
| 실제 새 환경 smoke test | 1.0 | 1.0 | 임시 Python 3.12 venv에서 editable install 후 5개 테스트와 config validation이 통과했다. |
| 저장소 위생 | 1.0 | 1.0 | 캐시·가상환경·대형 outputs/data/checkpoints ignore 규칙을 추가하고 테스트 임시 산출물을 제거 대상으로 식별했다. |

## 테스트 증거

```text
Successfully installed robust-vla-recovery-0.1.0
valid: project=robust-vla-recovery seed=20260903 device=auto tasks=4
Ran 5 tests in 0.028s
OK
```

## 설계 평가

### 잘 된 점

- 무거운 ML 패키지를 코어 패키지 import 경로에서 분리해 CI와 설정 검증이 GPU
  없이도 빠르게 실행된다.
- unknown key를 허용하지 않아 설정 오타가 조용히 무시되지 않는다.
- output path의 절대경로와 `..` 이탈을 막아 실행 결과가 프로젝트 밖의 임의
  위치에 기록되는 것을 예방한다.
- 범위 축소 규칙에 맞춰 config는 네 태스크뿐 아니라 세 태스크 구성도 허용한다.
- 최신 SmolVLA와 LIBERO 통합을 하나의 LeRobot 릴리스로 맞춰 오래된 LIBERO
  원본 환경과의 직접 충돌을 피했다.

### 남은 위험

1. 현재 작업 세션에서는 WSL 배포판 조회 권한이 없어 Linux 환경 존재 여부를
   검증하지 못했다.
2. simulation requirements는 직접 의존성을 정확히 고정했지만 전이 의존성 전체
   lock은 아직 생성되지 않았다.
3. 6GB VRAM에서 SmolVLA inference가 실제로 동작하는지는 Day 3~6의 설치 및
   짧은 rollout으로 확인해야 한다.
4. CI workflow는 작성하고 로컬 동등 명령을 통과했지만 GitHub 원격 실행 결과는
   아직 없다.

## Day 2 Gate

- [x] 표준 디렉터리 구조가 있다.
- [x] Python 버전과 핵심 simulation dependency 버전이 고정돼 있다.
- [x] seed, device, output directory가 공통 config에 있다.
- [x] import test와 config validation이 있다.
- [x] 새 임시 환경에서 설치와 smoke test가 통과했다.
- [x] 생성된 임시 환경과 캐시는 최종 저장소에서 제거한다.

## 다음 판단 지점

Day 3 시작 시 WSL2/Linux 환경 접근, NVIDIA CUDA 전달, `hf-libero` import를 먼저
확인한다. 이 세 항목 중 하나라도 실패하면 전체 패키지 재설치를 반복하지 말고
환경 문제를 먼저 고립해 해결한다.
