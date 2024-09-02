@echo off

echo Training Disc Random Single with SAC
python train_drone_algo_rllib.py -t disc -c -a sac -i 40 -w random_single --tuned --no-notif --max-steps 300

@REM echo Training Cont Random Single with SAC
@REM python train_drone_algo_rllib.py -t cont -c -a sac -i 40 -w random_single --tuned --no-notif --max-steps 300

@REM echo Training Disc Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t disc -c -a sac -i 40 -w random_multiple --tuned --no-notif --max-steps 500

@REM echo Training Cont Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t cont -c -a sac -i 40 -w random_multiple --tuned --no-notif --max-steps 500
