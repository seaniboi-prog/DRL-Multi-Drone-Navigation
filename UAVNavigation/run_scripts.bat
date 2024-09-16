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

@REM echo Single UAV Experiment Multiple with PPO
@REM python single_uav_experiment.py -a ppo -i 5 -w random_multiple -v multiple --best

@REM echo Single UAV Experiment Obstacle with PPO
@REM python single_uav_experiment.py -a ppo -i 5 -w random_multiple -v obstacle --best

@REM echo Single UAV Experiment Single with MARWIL
@REM python single_uav_experiment.py -a marwil -i 5 -w random_single -v single --best

echo Single UAV Experiment Multiple with MARWIL
python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v multiple --best --max-steps 300

echo Single UAV Experiment Obstacle with MARWIL
python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v obstacle --best --max-steps 300

echo Single UAV Experiment Single with SAC
python single_uav_experiment.py -a sac -i 5 -w random_single -v single --best --max-steps 300

echo Single UAV Experiment Multiple with SAC
python single_uav_experiment.py -a sac -i 5 -w random_multiple -v multiple --best --max-steps 300

echo Single UAV Experiment Obstacle with SAC
python single_uav_experiment.py -a sac -i 5 -w random_multiple -v obstacle --best --max-steps 300

@REM echo Training Disc Random Single with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_single --tuned --max-steps 300
@REM python restore_drone_algo_rllib.py -t disc -a sac -i 10 -w random_single

@REM echo Training Cont Random Single with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_single --tuned --max-steps 300

@REM echo Training Disc Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_multiple --tuned --max-steps 500

@REM echo Training Cont Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_multiple --tuned --max-steps 500

