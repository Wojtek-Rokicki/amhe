#!/bin/bash
echo "start algorithm experiments"

for j in $(seq 1 1 5)
do
	for p in $(seq 10 10 90)
	do
		echo "loop $j POPULATION SIZE: $p"
		python gym_ga.py -p $p -c 0.75 -m 0.75 -n 2 -s proportional -k averaging
	done

	for p in $(seq 100 100 1000)
	do
		echo "loop $j POPULATION SIZE: $p"
		python gym_ga.py -p $p -c 0.75 -m 0.75 -n 2 -s proportional -k averaging
	done
done
