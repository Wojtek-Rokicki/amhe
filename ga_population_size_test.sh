#!/bin/sh
: '
Runs tests for default (best) parameters of genetic 
algorithm to test population size influence.
'

echo "Start algorithm experiments"

for p in $(seq 10 10 90)
do
    echo "POPULATION SIZE: $p"
	python3 gym_ga.py -p $p
done

for p in $(seq 100 100 1000)
do
    echo "POPULATION SIZE: $p"
	python3 gym_ga.py -p $p
done