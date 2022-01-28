import genetic_algorithm.config as config
import gym
import genetic_algorithm.crossover as cross
import genetic_algorithm.generate_population as gen_pop
import genetic_algorithm.nn_forward as nn
import numpy as np
import random

import time

import genetic_algorithm as ga
import os

from option_parser import AppOptionParser
from save_results import save_algorithm_results, save_program_results

if __name__ == "__main__":
    start_time = time.time()
    # Genetic Algorithms parameters
    parser = AppOptionParser()
    (options, args) = parser.parse_args()

    GENERATIONS = config.GENERATIONS
    population_size = options.population_size
    Pc = options.crossover_rate
    Pm = options.mutation_rate
    mutation_standard_deviation = options.mutation_standard_deviation

    # Neural Network parameters
    input_size = config.INPUT_SIZE
    hidden_neurons = options.hidden_neurons

    # Preparing Gym Environment
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    obs = env.reset()  # obs holds the state variables [x xdot theta theta_dot]

    # Initialize parameters for GA
    chromosome_pool, chromosome_length = gen_pop.init_population(hidden_neurons,population_size)  # initial pool of chromosomes
    # chromosome_pool is a ndarray with dimension of (population, weights)

    fitness = np.zeros(population_size)
    reward = 0 # reward for each chromosome
    success_rewards_threshold = 1000  #the threshold reward
    success_num = 0 # if reward value exceeds threshold, then it counts as success
    solution_found = False # flag value informing if solution were found
    count_games = 0 # count number of played games

    #Initialize parameter for results file name
    parameters = f"{population_size}_{Pc}_{Pm}_{mutation_standard_deviation}_{hidden_neurons}_{options.selection}_{options.crossover}"

    # results
    best_fitness_list = []
    mean_fitness_list= []

    for generation in range(GENERATIONS):

        if solution_found:
                break

        # update chromosome pool via selection, crossover, and mutation
        if generation != 0: 

            #  1. selection
            if options.selection == 'proportional':
                selected_population = ga.proportional_selection(chromosome_pool, fitness)
            if options.selection == 'threshold':
                selected_population = ga.threshold_selection(chromosome_pool, fitness, 1/2)
            if options.selection == 'tournament':
                selected_population = ga.tournament_selection(chromosome_pool, fitness, 10)

            #  2. crossover
            if options.crossover == 'even':
                offspring = ga.even_crossover(selected_population, Pc)
            if options.crossover == 'averaging':
                offspring = ga.averaging_crossover(selected_population, Pc)
            if options.crossover == 'one_point':
                offspring = ga.one_point_crossover(selected_population, Pc)

            #  3. mutation
            chromosome_pool = ga.mutation(offspring, Pm, mutation_standard_deviation)

        print('Checking population results')
        for iteration in range(population_size):  # for every chromosome in population
            if solution_found:
                break
            
            observations = []
            actions = []
            rewards = []
            while True:  
                # produce a PROBABILITY of moving left or right for current chromosome and observation
                act = nn.nn_forward(obs, chromosome_pool[iteration],hidden_neurons)
                # corresponds to controlling the cartpole (>=0.5 +1 force applied)
                if act >= 0.5:
                    act = 1
                else:
                    act = 0

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)

                env.render()

                next_obs, reward, done, info = env.step(act)

                # save observations from the environment
                done = obs[0] < -2.4 \
                       or obs[0] > 2.4 \
                       or obs[2] < -45 * 2 * 3.14159 / 360 \
                       or obs[2] > 45 * 2 * 3.14159 / 360
                done = bool(done)

                if done: #if the game has failes
                    obs = env.reset()
                    print('Generation: ', generation)
                    print('Chromosome: ', iteration)
                    print('Fitness: ', sum(rewards))
                    reward = -1
                    fitness[iteration] = sum(rewards)
                    break
                else:
                    obs = next_obs
                
                # stop program if solution with success_rewards_threshold has been found
                if sum(rewards) >= success_rewards_threshold:
                    obs = env.reset()
                    print('Generation: ', generation)
                    print('Chromosome: ', iteration)
                    print('Fitness: ', sum(rewards))
                    reward = -1
                    fitness[iteration] = sum(rewards)
                    solution_found = True
                    break

            count_games +=1
        # save generation statistic to file
        best_fitness = np.amax(fitness)
        mean_fitness = np.mean(fitness)
        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)


    time = time.time() - start_time

    print(f'\n Program results')
    print(f'\n Time: {time}')
    print(f'\n Count games: {count_games} \n')

    #save programm results to file
    save_algorithm_results(parameters,count_games,time)
    save_program_results(parameters,best_fitness_list,mean_fitness_list)