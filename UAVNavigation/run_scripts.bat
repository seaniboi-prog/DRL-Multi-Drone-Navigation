@echo off

echo Restore Training Disc Random Multiple with PPO
python restore_drone_algo_rllib.py -t disc -a ppo -i 10 -w random_multiple

echo Restore Training Disc Random Multiple with PPO
python restore_drone_algo_rllib.py -t cont -a ppo -i 10 -w random_multiple