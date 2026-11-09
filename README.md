# CPG-RL · MuJoCo

Mini Cheetah용 MuJoCo 기반 CPG-RL/PPO 학습 환경. Python 3.10–3.13을 지원합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# 학습
cpg-train --num_envs 64 --headless

# 재개
cpg-train --resume --headless

# 재생 — model.pt를 실제 체크포인트 경로로 변경
cpg-play --checkpoint_path model.pt --headless

# 학습 로그
tensorboard --logdir logs
```

화면 재생은 `--headless`를 생략합니다. macOS에서는 `mjpython -m legged_gym.scripts.play --checkpoint_path model.pt`를 사용합니다.

체크포인트는 `logs/`에 저장됩니다. 재개·재생에는 같은 폴더의 `run_config.json`이 필요합니다. 설정은 `legged_gym/envs/mini_cheetah/config.py`, 추가 옵션은 각 명령의 `--help`를 확인하세요.

라이선스: `LICENSE`, `licenses/`. 로봇 자산 출처·관성 보정 내역: `resources/robots/provenance.json` (자산 재배포 조건은 별도 확인 필요).
