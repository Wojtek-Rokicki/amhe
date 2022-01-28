#!/bin/sh
: '
Runs QLearning algorithm 10 times.
'
for i in $(seq 1 1 10)
do
    python3 gym_ql.py
done