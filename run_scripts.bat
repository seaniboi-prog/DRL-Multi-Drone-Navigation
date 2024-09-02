@echo off

echo 5 Drone - 20 Random Nodes
python mtsp_experiments.py -d 5 -r 20

echo 5 Drone - 50 Random Nodes
python mtsp_experiments.py -d 5 -r 50

echo 5 Drone - 100 Random Nodes
python mtsp_experiments.py -d 5 -r 100

echo 5 Drone - 200 Random Nodes
python mtsp_experiments.py -d 5 -r 200