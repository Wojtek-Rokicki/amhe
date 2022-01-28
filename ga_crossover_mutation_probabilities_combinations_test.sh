#!/bin/sh
: '
Runs tests for combinations of crossover and mutation probability rates, 
for proportional selection and averaging crossover.
It uses default parameters for algorithm such as:
- maximum generations - 1000
- minimum rewards for solution - 1000
- population - 10
- mutation as normal standarized distribution
- one layer, two neurons perceptron neural net
'

echo "Start algorithm experiments - selection and crossover rates"

for i in $(seq 1 1 5)
do
    echo "Loop: $i  --crossover-rate 0.75 -m 0.75 "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.75 -m 0.75

    echo "Loop: $i  --crossover-rate 0.75 -m 0.5 "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.75 -m 0.5

    echo "Loop: $i  --crossover-rate 0.25 -m 0.75 "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.25 -m 0.75

    echo "Loop: $i  --crossover-rate 0.25 -m 0.25 "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.25 -m 0.25

    echo "Loop: $i  --crossover-rate 0.5 -m 0.25  "
	python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.5 -m 0.25

    echo "Loop: $i  --crossover-rate 0.5 -m 0.75  "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.5 -m 0.75

    echo "Loop: $i  --crossover-rate 0.25 -m 0.5  "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.25 -m 0.5

    echo "Loop: $i  --crossover-rate 0.75 -m 0.25 "
    python3 gym_ga.py -s proportional -c averaging --crossover-rate 0.75 -m 0.25


done
