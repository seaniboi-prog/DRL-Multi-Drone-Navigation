@echo off

echo Training Disc Random Single with MARWIL
python train_drone_algo_rllib.py -t disc -a marwil -i 40 -w random_single --tuned --max-steps 300

echo Training Cont Random Single with MARWIL
python train_drone_algo_rllib.py -t cont -a marwil -i 40 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with MARWIL
python train_drone_algo_rllib.py -t disc -a marwil -i 40 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Multiple with MARWIL
python train_drone_algo_rllib.py -t cont -a marwil -i 40 -w random_multiple --tuned --max-steps 500

echo Training Disc Random Single with SAC
python train_drone_algo_rllib.py -t disc -a sac -i 40 -w random_single --tuned --max-steps 300

echo Training Cont Random Single with SAC
python train_drone_algo_rllib.py -t cont -a sac -i 40 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with SAC
python train_drone_algo_rllib.py -t disc -a sac -i 40 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Multiple with SAC
python train_drone_algo_rllib.py -t cont -a sac -i 40 -w random_multiple --tuned --max-steps 500
