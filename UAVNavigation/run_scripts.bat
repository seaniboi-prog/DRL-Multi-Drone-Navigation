@echo off

@REM echo Tuning Cust (discrete) Random Single with SAC
@REM python tune_drone_algo_rllib.py -t disc --custom -a sac -i 40 -p 4 -s 30 -b 4096 -w random_single --max-steps 400

@REM echo Tuning Cust (discrete) Random Multiple with SAC
@REM python tune_drone_algo_rllib.py -t disc --custom -a sac -i 40 -p 4 -s 30 -b 4096 -w random_multiple --max-steps 500

echo Tuning Cust (continuous) Random Single with SAC
python tune_drone_algo_rllib.py -t cont --custom -a sac -i 40 -p 4 -s 30 -b 4096 -w random_single --max-steps 400 --no-notif

echo Tuning Cust (continuous) Random Multiple with SAC
python tune_drone_algo_rllib.py -t cont --custom -a sac -i 40 -p 4 -s 30 -b 4096 -w random_multiple --max-steps 500 --no-notif