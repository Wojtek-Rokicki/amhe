from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple

# import gym 
import gym

import os

env = gym.make('CartPole-v0')

n_bins = ( 6 , 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape

def policy( state : tuple ):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

# Adaptive learning of Learning Rate
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

n_episodes = 1000 
rewards = 0
rewards_threshold = 1000
solution_found = False

start_time = time.time()
stop_time = -1
for e in range(n_episodes):

    print(f'Episode no.: {e}')
    
    # Discretize state into buckets
    obs = env.reset()
    current_state, done = discretizer(*obs), False
    
    while done==False:
        
        # policy action 
        action = policy(current_state) # exploit
        
        # insert random action
        if np.random.random() < exploration_rate(e) : 
            action = env.action_space.sample() # explore 
         
        # increment enviroment
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)
        
        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward , new_state )
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
        
        current_state = new_state
        
        rewards = rewards + reward

        # if done and rewards>=500:
        # print("Done, but why ...")

        if rewards >= rewards_threshold:
            print("Solution found")
            solution_found = True
            break

        # Render the cartpole environment
        env.render()

        done = obs[0] < -2.4 \
                or obs[0] > 2.4 \
                or obs[2] < -45 * 2 * 3.14159 / 360 \
                or obs[2] > 45 * 2 * 3.14159 / 360
        done = bool(done)

        if done:
            env.reset()
        
    print(f'Rewards: {rewards}')
    rewards = 0

    if solution_found:
        games = e
        break
if not solution_found:
    games = n_episodes

stop_time = time.time()

duration_time = stop_time-start_time
# write general statistic to file
f = open("results/ql.csv", "a")
f.write(f'{games},{start_time},{stop_time},{duration_time}\n')
f.close()
            