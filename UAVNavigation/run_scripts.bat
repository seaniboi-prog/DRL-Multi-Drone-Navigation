@echo off

echo Tuning Cust Env1 (discrete) Random Multiple with PPO
python tune_drone_algo_rllib.py -t cust -e v1 -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_multiple --no-notif --max-steps 500

echo Tuning Cust Env2 (continuous) Random Single with PPO
python tune_drone_algo_rllib.py -t cust -e v2 -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_single --no-notif --max-steps 300

echo Tuning Cust Env2 (continuous) Random Multiple with PPO
python tune_drone_algo_rllib.py -t cust -e v2 -a ppo -i 40 -p 4 -s 30 -b 4096 -w random_multiple --no-notif --max-steps 500