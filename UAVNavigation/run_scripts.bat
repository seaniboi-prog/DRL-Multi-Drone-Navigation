@echo off

@REM echo Training Disc Random Single with MARWIL
@REM python restore_drone_algo_rllib.py -t disc -a marwil -i 4 -w random_single

@REM echo Training Cont Random Single with MARWIL
@REM python train_drone_algo_rllib.py -t cont -a marwil -i 35 -w random_single --tuned --max-steps 300

@REM echo Training Disc Random Multiple with MARWIL
@REM python train_drone_algo_rllib.py -t disc -a marwil -i 35 -w random_multiple --tuned --max-steps 500

@REM echo Training Cont Random Multiple with MARWIL
@REM python train_drone_algo_rllib.py -t cont -a marwil -i 35 -w random_multiple --tuned --max-steps 500


@REM echo Single UAV Experiment Single with PPO
@REM python single_uav_experiment.py -a ppo -i 5 -w random_single -v single --best

echo Single UAV Experiment Multiple with PPO
python single_uav_experiment.py -a ppo -i 5 -w random_multiple -v multiple --best

echo Single UAV Experiment Obstacle with PPO
python single_uav_experiment.py -a ppo -i 5 -w random_multiple -v obstacle --best

echo Single UAV Experiment Single with MARWIL
python single_uav_experiment.py -a marwil -i 5 -w random_single -v single --best

echo Single UAV Experiment Multiple with MARWIL
python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v multiple --best

echo Single UAV Experiment Obstacle with MARWIL
python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v obstacle --best

echo Training Disc Random Single with SAC
python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_single --tuned --max-steps 300

echo Training Cont Random Single with SAC
python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with SAC
python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Multiple with SAC
python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_multiple --tuned --max-steps 500

