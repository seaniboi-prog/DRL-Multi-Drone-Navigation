@echo off

echo Training Disc Random Multiple with PPO
python train_drone_algo_rllib.py -t disc -a ppo -i 40 -b 2048 -w random_multiple --tuned --max-steps 800
