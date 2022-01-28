#!/bin/bash
: '
Runs DQN algorithm 10 times.
'
for i in $(seq 1 1 10)
do
    python3 gym_dqn.py
done