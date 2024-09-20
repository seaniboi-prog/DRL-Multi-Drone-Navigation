@echo off

echo Training Disc Random Single with PPO
python train_drone_algo_rllib.py -t disc -a ppo -i 40 -w random_single --tuned --max-steps 300

echo Training Cont Random Single with PPO
python train_drone_algo_rllib.py -t cont -a ppo -i 40 -w random_single --tuned --max-steps 300

echo Training Disc Random Multiple with PPO
python train_drone_algo_rllib.py -t disc -a ppo -i 40 -w random_multiple --tuned --max-steps 500

echo Training Cont Random Multiple with PPO
python train_drone_algo_rllib.py -t cont -a ppo -i 40 -w random_multiple --tuned --max-steps 500

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


@REM echo Single UAV Experiment Single with PPO
@REM python single_uav_experiment.py -a ppo -i 10 -w random_single -v single --best -p

@REM echo Single UAV Experiment Multiple with PPO
@REM python single_uav_experiment.py -a ppo -i 10 -w random_multiple -v multiple --best -p

@REM echo Single UAV Experiment Obstacle with PPO
@REM python single_uav_experiment.py -a ppo -i 10 -w random_multiple -v obstacle --best -p

@REM echo Single UAV Experiment Single with MARWIL
@REM python single_uav_experiment.py -a marwil -i 5 -w random_single -v single --best

@REM echo Single UAV Experiment Multiple with MARWIL
@REM python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v multiple --best --max-steps 300

@REM echo Single UAV Experiment Obstacle with MARWIL
@REM python single_uav_experiment.py -a marwil -i 5 -w random_multiple -v obstacle --best --max-steps 300

@REM echo Single UAV Experiment Single with SAC
@REM python single_uav_experiment.py -a sac -i 5 -w random_single -v single --best --max-steps 300

@REM echo Single UAV Experiment Multiple with SAC
@REM python single_uav_experiment.py -a sac -i 5 -w random_multiple -v multiple --best --max-steps 300

@REM echo Single UAV Experiment Obstacle with SAC
@REM python single_uav_experiment.py -a sac -i 5 -w random_multiple -v obstacle --best --max-steps 300

@REM echo Training Disc Random Single with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_single --tuned --max-steps 300
@REM python restore_drone_algo_rllib.py -t disc -a sac -i 10 -w random_single

@REM echo Training Cont Random Single with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_single --tuned --max-steps 300

@REM echo Training Disc Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t disc -a sac -i 30 -w random_multiple --tuned --max-steps 500

@REM echo Training Cont Random Multiple with SAC
@REM python train_drone_algo_rllib.py -t cont -a sac -i 30 -w random_multiple --tuned --max-steps 500

