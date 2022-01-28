#!/bin/sh
: '
Runs tests for algorithms comparison (GA, DQN, QLearning).
It will measure number of games and time duration taken to find the solution.
Solution is the model which will gets 1000 rewards. 
Algorithms run until they find solution.
'
(cd dqn ; ./dqn_test.sh) && (cd ql ; ./ql_test.sh)

for p in $(seq 1 1 5)
do
    echo "loop: $i "
    python3 gym_ga.py -p 50 -s proportional -c averaging --crossover-rate 0.75 -m 0.75 -n 2
done