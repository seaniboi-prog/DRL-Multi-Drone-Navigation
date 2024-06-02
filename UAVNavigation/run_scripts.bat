@echo off

echo Tuning Cust (discrete) Random Single with PPO
python tune_drone_algo_rllib.py -t disc --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_single --no-notif --max-steps 300

echo Tuning Cust (discrete) Random Multiple with PPO
python tune_drone_algo_rllib.py -t disc --custom -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_multiple --no-notif --max-steps 500