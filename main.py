import genetic_algorithm.config as config
# from genetic_algorithm import config
import gym
import genetic_algorithm.crossover as cross
import genetic_algorithm.generate_population as gen_pop
import genetic_algorithm.nn_forward as nn
import numpy as np

def main():
    print("Hello World!")
    print(f'generations: {config.GENERATIONS}')

if __name__ == "__main__":
    main()
    GENERATIONS = config.GENERATIONS
    hidden_neurons = config.hidden_neurons
    input_size = config.input_size
    population_size = config.population_size
    Pc = config.Pc
    Pm = config.Pm
    crossed_parents = config.crossed_parents

    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    obs = env.reset()  # obs holds the state variables [x xdot theta theta_dot]
    # might need to scale data so doesn't saturate the neurons, can use z-scaling or other,
    # makes sure the data has zero mean and unit variance
    reward = 0
    success_num = 0
    fitness = np.zeros(population_size)

    chromosome_pool, chromosome_length = gen_pop.generate_population()  # initial pool of chromosomes

    for generation in range(GENERATIONS):

        # update mating pool via selection, crossover, and mutation
        # keep the best X chromosomes from the previous generation
        if generation != 0:  # i.e. this is not the first initial population

            #  1. evaluate fitness, cumulative probability
            total_fitness = sum(fitness)

            individual_fitness = fitness / total_fitness
            cp = np.cumsum(individual_fitness)  # cumulative probabilities

            #  2. selection
            r = np.random.uniform(0, 1, population_size)  # create list of random numbers for selection

            a = np.array([0])
            # a.shape = (1, 1)
            cp_augmented = np.concatenate((a, cp), axis=0)
            hist, bin_edges = np.histogram(r, cp_augmented)
            selected_population = []
            empty = 1

            for i in range(population_size):
                for j in range(hist[i]):
                    if empty == 1:
                        selected_population = chromosome_pool[i][:]
                        empty = 0
                    else:
                        selected_population = np.vstack((selected_population, chromosome_pool[i][:]))

            # might be a good thing to check if keeping the top 4 or 2 works better

            #  3. crossover
            # r = np.random.uniform(0, 1, population_size)  # create list of random numbers for crossover

            indices = np.argwhere(r > Pc)
            parent_count = 0
            empty = 1
            i = 0
            crossover_pairs = []

            while i < population_size:
                #TODO: czy można tu jakąś lepszą metodę brania random dać?
                r_crossover = np.random.rand(1)
                if r_crossover < Pc:
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
                if i == (population_size) and np.size(crossover_pairs, 0) < crossed_parents:
                    i = 0  # we don't have enough yet, so restart the loop
                    #  generate new random number list
                    r = np.random.uniform(0, 1, population_size)  # create list of random numbers for crossover

            offspring = cross.crossover(crossover_pairs)

            #  4. mutation
            # r =  np.random.uniform(0, 1, population_size*(hidden_neurons*input_size + hidden_neurons))
            print(f'Reshape: {offspring}')
            offspring.shape = (1, population_size*(chromosome_length ))

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
                z = sum(rewards)

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

            if sum(rewards) >= 300:
                success_num += 1
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
                    fitness[iteration] = sum(rewards)
                    break
            else:
                success_num = 0

