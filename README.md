# CPG-RL · MuJoCo

![Mini Cheetah와 CPG 상태를 보여주는 Viser 학습 모니터](viser-figure.png)

Mini Cheetah용 MuJoCo 기반 CPG-RL/PPO 학습 환경. Python 3.10–3.13을 지원합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# 학습 + Viser 웹 모니터 (http://127.0.0.1:8080)
cpg-train --num_envs 64

# 재개
cpg-train --resume

# 재생 — model.pt를 실제 체크포인트 경로로 변경
cpg-play --checkpoint_path model.pt --headless

# 학습 로그
tensorboard --logdir logs
```

학습 시 Viser에서 로봇·속도·보상·CPG 상태를 확인할 수 있습니다. `--headless`로 끄고, `--viewer_port 8081`로 포트를 변경합니다. 화면 재생은 `--headless`를 생략합니다. macOS에서는 `mjpython -m legged_gym.scripts.play --checkpoint_path model.pt`를 사용합니다.

체크포인트는 `logs/`에 저장됩니다. 재개·재생에는 같은 폴더의 `run_config.json`이 필요합니다. 설정은 `legged_gym/envs/mini_cheetah/config.py`, 추가 옵션은 각 명령의 `--help`를 확인하세요.

라이선스: `LICENSE`, `licenses/`. 로봇 자산 출처·관성 보정 내역: `resources/robots/provenance.json` (자산 재배포 조건은 별도 확인 필요).
