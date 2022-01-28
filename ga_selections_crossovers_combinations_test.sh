#!/bin/sh
: '
Runs tests for all combinations of selection and crossover methods.
It uses default parameters for algorithm such as:
- maximum generations - 1000
- minimum rewards for solution - 1000
- population - 10
- crossover probablility - 0.5
- mutation probability - 0.5
- mutation as normal standarized distribution
- one layer, two neurons perceptron neural net
'

echo "Start algorithm experiments - selection and crossover combinations"

for i in $(seq 1 1 5)
do
    echo "Loop: $i  selection: proportional crossover: even"
	python3 gym_ga.py -s proportional -c even

    echo "Loop: $i  selection: proportional crossover: averaging"
	python3 gym_ga.py -s proportional -c averaging

    echo "Loop: $i  selection: proportional crossover: one_point"
	python3 gym_ga.py -s proportional -c one_point


    echo "Loop: $i  selection: threshold crossover: even"
	python3 gym_ga.py -s threshold -c even

    echo "Loop: $i  selection: threshold crossover: averaging"
	python3 gym_ga.py -s threshold -c averaging

    echo "Loop: $i  selection: threshold crossover: one_point"
	python3 gym_ga.py -s threshold -c one_point


    echo "Loop: $i  selection: tournament crossover: even"
	python3 gym_ga.py -s tournament -c even

    echo "Loop: $i  selection: tournament crossover: averaging"
	python3 gym_ga.py -s tournament -c averaging

    echo "Loop: $i  selection: tournament crossover: one_point"
	python3 gym_ga.py -s tournament -c one_point

done
