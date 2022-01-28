#!/bin/bash
echo "start algorithm experiments - selection and crossover rates"


for i in $(seq 1 1 5)
do

    echo "loop: $i  -c 0.75 -m 0.25 "
    python gym_ga.py -s proportional -k averaging -c 0.75 -m 0.25

    echo "loop: $i  -c 0.75 -m 0.75 "
    python gym_ga.py -s proportional -k averaging -c 0.75 -m 0.75

    echo "loop: $i  -c 0.75 -m 0.5 "
    python gym_ga.py -s proportional -k averaging -c 0.75 -m 0.5

    echo "loop: $i  -c 0.25 -m 0.75 "
    python gym_ga.py -s proportional -k averaging -c 0.25 -m 0.75

    echo "loop: $i  -c 0.25 -m 0.25 "
    python gym_ga.py -s proportional -k averaging -c 0.25 -m 0.25

    echo "loop: $i  -c 0.5 -m 0.25  "
	python gym_ga.py -s proportional -k averaging -c 0.5 -m 0.25

    echo "loop: $i  -c 0.5 -m 0.75  "
    python gym_ga.py -s proportional -k averaging -c 0.5 -m 0.75

    echo "loop: $i  -c 0.25 -m 0.5  "
    python gym_ga.py -s proportional -k averaging -c 0.25 -m 0.5
    
    echo "loop: $i  -c 0.5 -m 0.5  "
    python gym_ga.py -s proportional -k averaging -c 0.5 -m 0.5


done
