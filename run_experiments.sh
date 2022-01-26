#!/bin/bash
echo "start algorithm experiments"

for p in $(seq 10 10 100)
do
    echo "POPULATION SIZE: $p"
	python gym_ga.py -p $p -c 0.5 -m 0.5 -v 1 -n 2
done

for p in $(seq 200 100 1000)
do
    echo "POPULATION SIZE: $p"
	python gym_ga.py -p $p -c 0.5 -m 0.5 -v 1 -n 2
done

for p in $(seq 1000 10000 1000)
do
    echo "POPULATION SIZE: $p"
	python gym_ga.py -p $p -c 0.5 -m 0.5 -v 1 -n 2
done