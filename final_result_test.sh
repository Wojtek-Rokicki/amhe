#!/bin/bash
echo "start algorithm run"

for p in $(seq 1 1 5)
do
    echo "loop: $i "
    python gym_ga.py -s proportional -k averaging -c 0.75 -m 0.75 -n 2 -p 50
done
