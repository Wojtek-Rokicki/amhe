import genetic_algorithm.config as config
import gym
import genetic_algorithm.crossover as cross
import genetic_algorithm.generate_population as gen_pop
import genetic_algorithm.nn_forward as nn
import numpy as np
import random

import time

import genetic_algorithm as ga

from option_parser import AppOptionParser

if __name__ == "__main__":
    start_time = time.time()
    # Genetic Algorithms parameters
    parser = AppOptionParser()
    (options, args) = parser.parse_args()
    GENERATIONS = config.GENERATIONS
    population_size = options.population_size
    Pc = options.crossover_rate
    Pm = options.mutation_rate
    mutation_variation = options.mutation_variation

    # Neural Network parameters
    input_size = config.INPUT_SIZE
    hidden_neurons = options.hidden_neurons

    # Preparing Gym Environment
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    obs = env.reset()  # obs holds the state variables [x xdot theta theta_dot]
    # might need to scale data so doesn't saturate the neurons, can use z-scaling or other,
    # makes sure the data has zero mean and unit variance

    # Initialize parameters for GA
    chromosome_pool, chromosome_length = gen_pop.init_population(hidden_neurons,population_size)  # initial pool of chromosomes
    # chromosome_pool is a ndarray with dimension of (population, weights)
    # chromosome_length is redundant information ...
    fitness = np.zeros(population_size)
    reward = 0 # reward for each chromosome
    success_rewards_threshold = 1000
    success_num = 0 # if reward value exceeds threshold, then it counts as success
    solution_found = False # flag value informing if solution were found
    count_games = 0 # count number of played games

    #Initialize parameter for results file name
    parameters = f"{population_size}_{Pc}_{Pm}_{mutation_variation}_{hidden_neurons}"
    f = open(f"program_results_{parameters}.csv", "a")
    f.write(f'generation,best_fitness,mean_fitness\n')
    f.close()

    for generation in range(GENERATIONS):

        if solution_found:
                break

        # update mating pool via selection, crossover, and mutation
        # keep the best X chromosomes from the previous generation
        if generation != 0:  # i.e. this is not the first initial population

            #  1. selection
            selected_population = ga.proportional_selection(chromosome_pool, fitness)
            #selected_population = ga.threshold_selection(chromosome_pool, fitness, 1/2)
            #selected_population = ga.tournament_selection(chromosome_pool, fitness, 10)

            #  2. crossover
            #offspring = ga.averaging_crossover(selected_population, Pc)
            #offspring = ga.one_point_crossover(selected_population, Pc)
            offspring = ga.even_crossover(selected_population, Pc)

            #  3. mutation
            offspring = ga.mutation(offspring, Pm, mutation_variation)

            #  5. new mating pool finalized
            chromosome_pool = offspring

        print('Checking population results')
        for iteration in range(population_size):  # episode
            if solution_found:
                break
            
            observations = []
            actions = []
            rewards = []
            while True:  # run each action which is much less than episode length
                # function to determine correct action given observation
                # it will only produce a PROBABILITY of moving left or right, this is a STOCHASTIC policy
                # we will then sample from this distribution using random # [0,1]
                act = nn.nn_forward(obs, chromosome_pool[iteration],hidden_neurons)  # current chromosome in the generation
                # corresponds to controlling the cartpole (>=0.5 +1 force applied)
                if act >= 0.5:
                    act = 1
                else:
                    act = 0
                # act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)

                env.render()

                next_obs, reward, done, info = env.step(act)
                z = sum(rewards) # for what?

                # TODO: Do environment configuration file, probably add some rewards threshold ...
                done = obs[0] < -2.4 \
                       or obs[0] > 2.4 \
                       or obs[2] < -45 * 2 * 3.14159 / 360 \
                       or obs[2] > 45 * 2 * 3.14159 / 360
                done = bool(done)

                if done:
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
        # write generation statistic to file
        f = open(f"program_results_{parameters}.csv", "a")
        best_fitness = np.amax(fitness)
        mean_fitness = np.mean(fitness)
        print(f"Best: {best_fitness}; mean: {mean_fitness}")
        f.write(f'{generation},{best_fitness},{mean_fitness}\n')
        f.close()
            #stop program if there are 100 chromosomes in popultaion with success_rewards_threshold fitness
            # if sum(rewards) >= success_rewards_threshold: # maybe break for other generations if condition is met?
            #     success_num += 1
            #     if success_num >= 100:
            #         print('Iteration: ', iteration)
            #         print('Clear!!')
            #         fitness[iteration] = sum(rewards)
            #         solution_found = True
            #         break
            # else:
            #     success_num = 0


    time = time.time() - start_time


    #print(f'\n Parameters: {parameters}')
    print(f'\n Time: {time}')
    print(f'\n Count games: {count_games} \n')

    # write general statistic to file
    f = open("algorithm_results.csv", "a")
    #f.write(toString(count_games)+","+time+"\n")
    f.write(f'{parameters},{count_games},{time}\n')
    f.close()
