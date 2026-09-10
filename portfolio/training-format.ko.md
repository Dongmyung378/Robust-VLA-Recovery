# 정책 학습 데이터 형식

[English](training-format.md) | [프로젝트 README](../README.ko.md)

## 목적

이 형식은 감사를 마친 LIBERO 데모를 정책 fine-tuning 입력으로 결정적으로 변환한다.
원본 카메라 pixel과 에피소드 경계를 보존하고, 수치형 변환을 명시하며, 언어와 frame의
관계를 직접 기록한다. 변환기는 에피소드 하나씩 처리하므로 전체 이미지 데이터를
메모리에 올리지 않는다.

## 원본과 분할

변환기는 `configs/data_audit.toml`에 고정된 파일 네 개로 Day 8 감사를 다시 생성한다.
byte 크기, SHA-256, dataset shape 또는 에피소드 개수가 달라지면 학습 데이터를 쓰기
전에 중단한다. split 순서는 seed `378`로 결정하며, 같은 초기 상태 group은 train,
validation, test 중 하나에만 들어간다.

변환 데이터는 에피소드 200개와 transition 26,145개다. 태스크마다 train 40개,
validation 5개, test 5개를 제공한다.

## HDF5 구성

```text
dataset.h5
  episode_lengths        int64[200]
  episode_offsets        int64[201]
  episodes/
    000000/
      observation/
        agent_image      uint8[T, 128, 128, 3]
        wrist_image      uint8[T, 128, 128, 3]
        state            float32[T, 15]
        frame_index      int64[T]
        timestamp_ns     int64[T]
        language_index   int32[T]
      action             float32[T, 7]
      language_token_ids int32[L]
      language_attention_mask uint8[L]
```

15차원 state는 `ee_states`, `gripper_states`, `joint_states` 순서로 결합한다. RGB 배열은
byte 단위로 그대로 복사하고 state와 action은 `float32`로 변환한다. 큰 배열에는 무손실
gzip level 1, shuffle, Fletcher32와 결정적 chunk를 사용한다.

`episode_offsets[i]`는 에피소드 `i`의 전역 시작 transition이며 마지막 offset은
26,145다. 각 에피소드 group에는 task, source file, source episode, split, scene, 초기
상태 split group, instruction과 source seed 유무도 기록한다.

## 시간과 언어 정렬

frame `t`와 action `t`가 정책 학습 row 하나를 이룬다. `frame_index`는 0부터 `T-1`까지,
`timestamp_ns[t]`는 20 Hz 기준 `t * 50,000,000`이다. `language_index`는 모두 0이며,
에피소드의 모든 frame을 하나의 instruction과 token sequence에 연결한다.

검사용 `lowercase-wordpunct-v1` tokenizer는 Unicode NFKC 정규화 후 소문자로 바꾸고
단어와 문장 부호를 분리한다. vocabulary는 train instruction만 사용해 만들고 `<pad>`,
`<bos>`, `<eos>`, `<unk>` ID를 고정한다. 이 tokenizer는 데이터 정렬과 vocabulary
출처를 검사하기 위한 것이다. Day 10에서는 instruction 원문을 유지한 채 선택한 정책의
tokenizer로 매핑할 수 있다.

## 검증 결과

전체 변환을 두 번 실행한 HDF5와 보고서의 체크섬이 각각 일치했다. 보존한 HDF5는
1,192,042,683 byte이며 SHA-256은
`f8ec588217d3c5a19f350394d01f428d9380efd1ae1e2e74a53ca2d3ff082da0`이다.

검증기는 고정된 원본에서 200개 split 배정을 다시 만들고 episode offset, array shape,
dtype, 유한값, timestamp, language index, token, metadata와 전체 파일 체크섬을 검사했다.
모든 태스크와 세 split을 포함하도록 표본 10개를 결정적으로 선택했고, 두 카메라 배열,
state와 action이 해당 원본과 일치했다. 로컬에 보존한 접촉 시트에서도 두 카메라의
원본과 변환 frame이 같은 것을 확인했다.

## 명령

```bash
rvla-convert-demos plan
rvla-convert-demos run
rvla-convert-demos verify outputs/day09/<run>/conversion.json
rvla-convert-demos sample-check outputs/day09/<run>/conversion.json --count 10
rvla-convert-demos repro-check outputs/day09/<run-a>/conversion.json \
  outputs/day09/<run-b>/conversion.json
```

원본 데이터, 변환 HDF5, manifest와 수동 검토 이미지는 Git에서 제외된 `data/`와
`outputs/` 아래에만 둔다. Git에는 변환 코드, 고정 설정, 검사와 공개 형식 문서만 남긴다.
