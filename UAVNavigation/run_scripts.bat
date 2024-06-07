@echo off

echo Training Disc Random Multiple with PPO
python restore_drone_algo_rllib.py -t disc -a ppo -i 5 -w random_multiple