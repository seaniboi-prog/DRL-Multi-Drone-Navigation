@echo off

echo Training Cont Random Multiple with PPO

python restore_drone_algo_rllib.py -t cont -a ppo -i 6 -w random_multiple
