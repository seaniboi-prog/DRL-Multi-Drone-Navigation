@echo off

echo Training Cont Random Multiple with PPO
python train_drone_algo_rllib.py -t cont -a ppo -i 6 -b 4096 -w random_single --tuned --max-steps 300

python restore_drone_algo_rllib.py -t cont -a ppo -i 6 -w random_single

python restore_drone_algo_rllib.py -t cont -a ppo -i 6 -w random_single

python restore_drone_algo_rllib.py -t cont -a ppo -i 6 -w random_single

python restore_drone_algo_rllib.py -t cont -a ppo -i 6 -w random_single