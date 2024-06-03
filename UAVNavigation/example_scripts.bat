@echo off

echo Tuning Cust (discrete) Random Single with PPO
python tune_drone_algo_rllib.py -t disc --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_single --no-notif --max-steps 300

echo Tuning Cust (discrete) Random Multiple with PPO
python tune_drone_algo_rllib.py -t disc --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_multiple --no-notif --max-steps 500

echo Tuning Cust (continuous) Random Single with PPO
python tune_drone_algo_rllib.py -t cont --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_single --no-notif --max-steps 300

echo Tuning Cust (continuous) Random Multiple with PPO
python tune_drone_algo_rllib.py -t cont --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_multiple --no-notif --max-steps 500

echo Training Disc Random Single with PPO
python train_drone_algo_rllib.py -t disc -a ppo -i 30 -b 4096 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with PPO
python train_drone_algo_rllib.py -t disc -a ppo -i 40 -b 4096 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Single with PPO
python train_drone_algo_rllib.py -t cont -a ppo -i 30 -b 4096 -w random_single --tuned --max-steps 300

echo Training Cont Random Multiple with PPO
python train_drone_algo_rllib.py -t cont -a ppo -i 40 -b 4096 -w random_multiple --tuned --max-steps 500

echo Script execution completed.