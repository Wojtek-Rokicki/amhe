#!/bin/bash
for i in $(seq 1 1 10)
do
    python3 dqn/cartpole.py
done