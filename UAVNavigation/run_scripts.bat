@echo off

echo Training Disc Random Single with MARWIL
python restore_drone_algo_rllib.py -t disc -a marwil -i 4 -w random_single

echo Training Cont Random Single with MARWIL
python train_drone_algo_rllib.py -t cont -a marwil -i 35 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with MARWIL
python train_drone_algo_rllib.py -t disc -a marwil -i 35 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Multiple with MARWIL
python train_drone_algo_rllib.py -t cont -a marwil -i 35 -w random_multiple --tuned --max-steps 500

@REM echo Training Disc Random Single with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_single --tuned --max-steps 300

@REM echo Training Cont Random Single with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_single --tuned --max-steps 300

@REM echo Training Disc Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_multiple --tuned --max-steps 500

@REM echo Training Cont Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_multiple --tuned --max-steps 500
