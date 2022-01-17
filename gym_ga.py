import genetic_algorithm.config as config
import gym
import genetic_algorithm.crossover as cross
import genetic_algorithm.generate_population as gen_pop
import genetic_algorithm.nn_forward as nn
import numpy as np

import genetic_algorithm as ga

if __name__ == "__main__":

    # Genetic Algorithms parameters
    GENERATIONS = config.GENERATIONS
    population_size = config.population_size
    crossed_parents = config.crossed_parents
    Pc = config.crossover_rate
    Pm = config.mutation_rate

    # Neural Network parameters
    input_size = config.INPUT_SIZE
    hidden_neurons = config.hidden_neurons

    # Preparing Gym Environment
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    obs = env.reset()  # obs holds the state variables [x xdot theta theta_dot]
    # might need to scale data so doesn't saturate the neurons, can use z-scaling or other,
    # makes sure the data has zero mean and unit variance

    # Initialize parameters for GA
    chromosome_pool, chromosome_length = gen_pop.init_population()  # initial pool of chromosomes
    # chromosome_pool is a ndarray with dimension of (population, weights)
    # chromosome_length is redundant information ...
    fitness = np.zeros(population_size)
    reward = 0 # reward for each chromosome
    success_rewards_threshold = 300
    success_num = 0 # if reward value exceeds threshold, then it counts as success


    for generation in range(GENERATIONS):

        # update mating pool via selection, crossover, and mutation
        # keep the best X chromosomes from the previous generation
        if generation != 0:  # i.e. this is not the first initial population

            #  1. selection
            # selected_population = ga.proportional_selection(chromosome_pool, fitness)
            # selected_population = ga.threshold_selection(chromosome_pool, fitness, 1/2)
            selected_population = ga.tournament_selection(chromosome_pool, fitness, 10)

            #  2. crossover
            # TODO: Same as with selection ...
            r = np.random.uniform(0, 1, population_size)  # create list of random numbers for crossover

            indices = np.argwhere(r > Pc) # it is using same randomly choosen values (same as for selection)??? not used ...
            parent_count = 0
            empty = 1
            i = 0
            crossover_pairs = []

            while i < population_size:
                #TODO: czy można tu jakąś lepszą metodę brania random dać?
                r_crossover = np.random.rand(1)
                if r_crossover >= Pc:
                    # include ith chromosome
                    if empty == 1:
                        crossover_pairs = selected_population[i][:]
                        #TODO: zastąpić to 12 długością chromosomu
                        crossover_pairs.shape = (1, chromosome_length)
                        empty = 0
                    else:
                        crossover_pairs = np.vstack((crossover_pairs, selected_population[i][:]))

                    if np.size(crossover_pairs, 0) == crossed_parents:
                        break  # we have enough parents for crossover
                i = i + 1
                if i == (population_size) and np.size(crossover_pairs, 0) < crossed_parents: # TODO: Krzyzowanie dla tylko czesci osobnikow, reszta niezmieniona, wg wykladu Arabasa
                    i = 0  # we don't have enough yet, so restart the loop
                    #  generate new random number list
                    r = np.random.uniform(0, 1, population_size)  # create list of random numbers for crossover

            offspring = cross.crossover(crossover_pairs)

            #  3. mutation
            # r =  np.random.uniform(0, 1, population_size*(hidden_neurons*input_size + hidden_neurons))
            print(f'Reshape: {offspring}')
            offspring.shape = (1, population_size*(chromosome_length )) # tutaj blad wyskakuje ... tak, poniewaz rozmiary sie nie zgadzaly, zauwaz, ze w przykladzie rozmiar populacji jest rowny liczbie osobnikow do krzyzowania

            for m in range(np.size(r, 0)):
                r_mutation = np.random.rand(1)
                if r_mutation < Pm:
                    random_value = np.random.uniform(-1.0, 1.0, 1)
                    offspring[0][m] += random_value

            #TODO: zasßapić to 2 liczbą biasów w sieci
            offspring.shape = (population_size, chromosome_length)
            #  5. new mating pool finalized

            chromosome_pool = offspring

        print('Checking population results')
        for iteration in range(population_size):  # episode
            observations = []
            actions = []
            rewards = []
            while True:  # run each action which is much less than episode length

                # function to determine correct action given observation
                # it will only produce a PROBABILITY of moving left or right, this is a STOCHASTIC policy
                # we will then sample from this distribution using random # [0,1]
                act = nn.nn_forward(obs, chromosome_pool[iteration])  # current chromosome in the generation
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

            if sum(rewards) >= success_rewards_threshold: # maybe break for other generations if condition is met?
                success_num += 1
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
                    fitness[iteration] = sum(rewards)
                    break
            else:
                success_num = 0

