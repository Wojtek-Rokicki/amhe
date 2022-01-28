#!/bin/bash
echo "start algorithm experiments - selection and crossover combinations"

for i in $(seq 1 1 5)
do
    echo "loop: $i  selection: proportional crossover: even"
	python gym_ga.py -s proportional -k even

    echo "loop: $i  selection: proportional crossover: averaging"
	python gym_ga.py -s proportional -k averaging

    echo "loop: $i  selection: proportional crossover: one_point"
	python gym_ga.py -s proportional -k one_point


    echo "loop: $i  selection: threshold crossover: even"
	python gym_ga.py -s threshold -k even

    echo "loop: $i  selection: threshold crossover: averaging"
	python gym_ga.py -s threshold -k averaging

    echo "loop: $i  selection: threshold crossover: one_point"
	python gym_ga.py -s threshold -k one_point


    echo "loop: $i  selection: tournament crossover: even"
	python gym_ga.py -s tournament -k even

    echo "loop: $i  selection: tournament crossover: averaging"
	python gym_ga.py -s tournament -k averaging

    echo "loop: $i  selection: tournament crossover: one_point"
	python gym_ga.py -s tournament -k one_point

done
