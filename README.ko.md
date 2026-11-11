# CPG-RL & AllGaits · MuJoCo

[English](README.md) | **한국어**

![Mini Cheetah와 CPG 상태를 보여주는 Viser 학습 모니터](viser-figure.png)

**CPG-RL**과 **AllGaits**를 Mini Cheetah 사족보행 로봇에 적용한 MuJoCo/PPO 구현입니다. 물리 시뮬레이션은 CPU에서 실행하며, PPO 학습은 CPU·CUDA·MPS를 지원합니다. Python 3.10–3.13이 필요합니다.

| 논문 | Task | 구현 내용 |
|---|---|---|
| [CPG-RL (2022)](https://arxiv.org/abs/2211.00458) | `mini_cheetah_cpg` (기본값) | 다리별 진폭·주파수·방향 변화율의 12차원 행동. 기본 설정은 비결합 발진기입니다. |
| [AllGaits (2024)](https://arxiv.org/abs/2411.04787) | `mini_cheetah_allgaits` | 다리별 진폭·주파수의 8차원 행동. 9개 보행의 결합 행렬을 사용하며, 학습 중 3초마다 보행을 다시 선택합니다. |

두 방식 모두 **PPO → CPG → 발 궤적 → IK → 관절 PD 제어** 구조를 사용합니다. 비공식 구현으로, 로봇 기하·제어 게인·학습 범위·위상 인코딩은 원 논문과 다릅니다. 논문에 보고된 성능과 실물 로봇 실험까지 재현한 것은 아닙니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# CPG-RL 학습 + Viser (http://127.0.0.1:8080)
cpg-train --num_envs 64

# AllGaits 학습 + Viser
cpg-train --task mini_cheetah_allgaits --num_envs 64

# 최근 학습 재개 — 학습할 때 사용한 task 지정
cpg-train --task mini_cheetah_allgaits --resume

# 재생 — model.pt를 실제 체크포인트 경로로 변경
cpg-play --checkpoint_path model.pt --headless
cpg-play --task mini_cheetah_allgaits --checkpoint_path model.pt --gait trot --headless

# 학습 지표
tensorboard --logdir logs
```

Viser에서 로봇·보상·CPG 상태·AllGaits 결합 행렬을 확인할 수 있습니다. 주황 화살표는 명령 속도, 초록 화살표는 실제 속도입니다. 명령은 로봇 몸체 좌표계 기준이며, `Front (+X)`가 앞쪽을 표시합니다. `--headless`로 Viser를 끄거나 `--viewer_port 8081`로 포트를 변경할 수 있습니다.

네이티브 뷰어로 재생하려면 `--headless`를 생략합니다. macOS에서는 `cpg-play` 대신 `mjpython -m legged_gym.scripts.play`를 사용합니다. AllGaits 재생은 `--gait`로 보행을 고정하거나, `--gait random`으로 3초마다 보행을 다시 선택할 수 있습니다.

체크포인트는 `logs/`에 저장됩니다. 재개·재생에는 학습과 동일한 `--task` 및 체크포인트와 같은 폴더의 `run_config.json`이 필요합니다. 설정: [config.py](legged_gym/envs/mini_cheetah/config.py). 추가 옵션: `cpg-train --help`, `cpg-play --help`.

라이선스: [LICENSE](LICENSE), [licenses/](licenses/). 로봇 자산 출처·관성 보정·재배포 관련 사항: [provenance.json](resources/robots/provenance.json).

## AllGaits 기반 정책의 포화·방향 유지 검증

인코더를 추가하기 전에 기존 68차원 관측·8차원 행동·CPG를 유지한 채 원인을 분리합니다. 기본 설정은 기존 구현 그대로이며, `--allgaits_profile`은 **새 학습에만** 적용합니다.

| 프로필 | 직전 비교군에서 바뀌는 항목 |
|---|---|
| 옵션 생략 | 기존 Gaussian + entropy 0.01 |
| `entropy0` | entropy만 0으로 변경 |
| `bounded` | `entropy0`에서 행동 분포만 [-1, 1] Beta로 변경 |
| `bounded_yaw` | `bounded`에 `tracking_yaw` 보상 가중치 1.0 추가 |

Beta는 설치된 rsl-rl의 분포·log probability·KL·추론 export를 사용하며, 정책 출력 뒤에 tanh만 붙이는 방식이 아닙니다. 환경 행동은 여전히 8차원이지만 Beta는 분포 파라미터용 출력 head가 달라집니다. 이 프로필은 개선 여부를 검증할 실험 변형이며 원 논문 그대로의 재현이라는 의미는 아닙니다.

```bash
# 기존 model_200.pt와 같은 201회 업데이트. 기존 학습과 동일한 환경 수·seed 사용.
python -m legged_gym.scripts.train --task mini_cheetah_allgaits \
  --allgaits_profile entropy0 --observation full --gait random \
  --num_envs 4096 --num_threads 4 --seed 40 --rl_device cpu \
  --max_iterations 201 --experiment_name allgaits_entropy0 --headless

# 체크포인트 경로는 실제 학습 폴더에 맞게 지정합니다.
python -m legged_gym.scripts.evaluate_allgaits \
  --checkpoints /path/to/original/model_200.pt /path/to/entropy0/model_200.pt \
  --gaits walk trot --seeds 100 101 102 --num-envs 8 --steps 1000 \
  --out results/allgaits_entropy0_201
```

결과는 `comparison.csv`와 보행별 JSON에 저장됩니다. 실패율·속도 MAE와 함께 `action_clipped_fraction`(범위 밖 출력), `action_boundary_fraction`(범위 95% 이상), `mean_max_lateral_deviation_m`, `mean_max_heading_deviation_rad`, `mean_yaw_rate`를 비교합니다. 궤적은 초기 위치·방향을 기준으로 하며 준비 구간부터 첫 실패/종료까지 기록합니다. 회전 MAE는 흔들림과 방향 편향을 함께 포함하므로 평균 yaw rate·누적 방향 편차도 확인하세요. `corridor_exit_fraction`은 폭 ±1.4 m 이탈 진단이며 종료 조건이 아닙니다.

`entropy0`에서도 포화가 지속되면 같은 seed·학습량으로 `bounded`를 비교하고, 방향 문제가 남으면 `bounded_yaw`를 비교합니다. 다른 프로필은 기존 모델에서 resume하지 않고 새로 학습합니다. 같은 프로필의 학습을 이어갈 때는 `--allgaits_profile` 없이 `--resume --checkpoint_path ...`를 사용합니다. 기존 baseline runner의 재개는 optimizer/normalizer와 학습 카운터를 복원하지만 simulator/RNG를 완전히 복원하는 exact resume는 아닙니다. 후보 확정 후 30초 평가(`--steps 3000`), 9개 보행 및 `random` 전환을 확인합니다. 모든 후보에 같은 평가 조건을 적용하세요.

## CPG–Reflex 협력 학습 (실험용)

`mini_cheetah_coop_allgaits`는 frozen AllGaits(100 Hz), 느린 CPG residual(기본 20 Hz), 빠른 발 위치/위상 residual(100 Hz)을 결합합니다. 기존 단일 PPO와 분리된 runner를 사용합니다. 현재 제공하는 것은 학습·평가 구현이며, 논문 수준의 보행 성능이나 실물 전이는 아직 검증하지 않았습니다.

먼저 기존 명령으로 **AllGaits** 기반 정책을 학습해야 합니다. `mini_cheetah_cpg`의 84차원 체크포인트는 사용할 수 없습니다. 기반 체크포인트와 같은 폴더의 `run_config.json`을 함께 보관하세요.

```bash
# BASE는 실제 AllGaits 체크포인트 경로로 설정합니다.
BASE=/path/to/allgaits/model.pt
python -m legged_gym.scripts.inspect_baseline --checkpoint "$BASE"
python -m legged_gym.scripts.profile_coop --base-checkpoint "$BASE" --device cpu --num-envs 32 64 128

# 소규모 실행 검증 후 iterations와 환경 수를 실측 예산에 맞게 늘립니다.
python -m legged_gym.scripts.train_coop --base-checkpoint "$BASE" \
  --config configs/coop_allgaits.json --stage reflex_only \
  --num-envs 64 --iterations 100 --device cpu --out logs/coop_r_seed40
python -m legged_gym.scripts.train_coop --base-checkpoint "$BASE" \
  --config configs/coop_allgaits.json --stage slow_only \
  --num-envs 64 --iterations 100 --device cpu --out logs/coop_s_seed40
python -m legged_gym.scripts.train_coop --base-checkpoint "$BASE" \
  --config configs/coop_allgaits.json --stage sequential \
  --reflex-checkpoint logs/coop_r_seed40/checkpoints/last.pt \
  --slow-checkpoint logs/coop_s_seed40/checkpoints/last.pt \
  --num-envs 64 --iterations 100 --out logs/coop_seq_seed40
python -m legged_gym.scripts.train_coop --base-checkpoint "$BASE" \
  --config configs/coop_allgaits.json --stage cooperative_spatial \
  --reflex-checkpoint logs/coop_r_seed40/checkpoints/last.pt \
  --slow-checkpoint logs/coop_seq_seed40/checkpoints/last.pt \
  --num-envs 64 --iterations 100 --out logs/coop_joint_seed40
```

`--device cpu|mps|auto`, `--seed`, `--rollout-steps`, `--terrain`, `--terrain-seed`, `--physics-dt`를 지원합니다. 작은 배치에서 MPS가 더 빠르다고 가정하지 말고 profile 결과로 선택하세요. 위 반복 횟수는 실행 예시이며 공정한 최종 연구 예산이 아닙니다. warm-start 단계의 데이터까지 합산하고 실제 `total_env_steps`와 wall-clock을 비교해야 합니다. `last.pt`는 마지막 상태이며 검증 세트에서 선택한 최적 모델이라는 의미가 아닙니다.

추가 stage는 `no_message`, `same_rate_100`, `same_rate_20`, `monolithic`, `cooperative_phase`입니다. `--resume`은 동일 설정·seed·환경 수·소스 코드·런타임 버전·학습 장치에서 모델, optimizer, normalizer, simulator, RNG 상태를 복구합니다. `--slow-checkpoint`/`--reflex-checkpoint`는 각 분기의 가중치를 가져오는 warm-start이며 exact resume와 구분합니다. 결정 주기가 다른 단일 정책도 같은 시점에만 행동을 샘플링합니다. 공유 actor의 용량은 기본 두 개의 256–256 actor에 근접하도록 정했으며, hidden 크기를 바꾸면 다시 파라미터 수를 비교하세요.

평가용 `models.json` 예:

```json
{
  "base_checkpoint": "/path/to/allgaits/model.pt",
  "models": [
    {"name": "base", "stage": "base", "training_seed": 0},
    {"name": "cooperative", "checkpoint": "logs/coop_joint_seed40/checkpoints/last.pt", "training_seed": 40},
    {"name": "independent", "stage": "independent_merge", "training_seed": 40,
     "slow_checkpoint": "logs/coop_s_seed40/checkpoints/last.pt",
     "reflex_checkpoint": "logs/coop_r_seed40/checkpoints/last.pt"}
  ]
}
```

```bash
python -m legged_gym.scripts.eval_coop --models models.json \
  --scenario-set configs/coop_scenarios.json --out results/paired --traces
pip install -e '.[analysis]'
python -m legged_gym.scripts.analyze_coop --episodes results/paired/metrics_per_episode.csv \
  --reference base --out results/paired/analysis --plots
python -m legged_gym.scripts.render_coop --base-checkpoint "$BASE" \
  --checkpoint logs/coop_joint_seed40/checkpoints/last.pt --out results/demo.mp4
python -m unittest discover -s tests -v
```

제공된 scenario 파일은 4개 사례의 시작 예시입니다. 최종 연구에는 독립적인 학습 seeds와 충분한 validation/test layout을 별도로 구성해야 합니다. 평가는 policy마다 동일한 manifest의 첫 episode를 사용하고, 완료율과 실패를 기록합니다. 기계적 COT는 1 m 이상 전진하여 course를 완료한 경우에만 계산하며, paired COT 분석은 양쪽 모두 완료한 사례로 조건화되므로 반드시 전체 완료율을 함께 해석하세요. bootstrap은 training seed를 상위 단위로 사용합니다. 단일 seed 결과의 구간은 학습 불확실성을 나타내지 않습니다.

Actor의 body velocity/contact는 **시뮬레이션 관측**이며, actor slip 채널은 현재 0입니다. `contact_delay_steps`는 10 ms 단위 지연, `actuator_delay_steps`는 physics tick 단위 명령 지연입니다. `contact_dropout`은 physics sample마다 적용됩니다. 질량·모터·마찰 랜덤화는 환경 생성 시, irregular 지형 랜덤화는 episode reset마다 적용됩니다. 접촉 센서는 지형이 활성화되면 floor 외의 표면도 측정합니다. 설정 변경과 seed는 모두 실험 결과와 함께 보관하세요.
