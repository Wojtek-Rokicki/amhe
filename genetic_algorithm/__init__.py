import numpy as np
import random

# TODO: for selection types arguments check: s, theta
def proportional_selection(p, f):
    '''Proportional selection for Evolutionary Algorithms
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       f : ndarray
            fitness function value of individuals
       
       Returns
       -------
       ndarray
            an array of selected individuals
    '''

    total_fitness = np.sum(f)
    ps = f / total_fitness # individuals selection probabilities
    cp = np.cumsum(ps) # cumulative probabilities
    cp_augmented = np.concatenate( ( np.array([0]), cp ), axis=0) # falling into proper bins will resamble proportional selection probablities

    mi = p.shape[0] # population size
    r = np.random.uniform(0, 1, mi) # create list of random numbers for selection
    hist, _ = np.histogram(r, cp_augmented)

    # Select each new individual
    selected_population = []
    empty = 1
    for i in range(mi):
        for _ in range(hist[i]):
            if empty:
                selected_population = p[i]
                empty = 0
            else:
                selected_population = np.vstack((selected_population, p[i]))

    return selected_population

def threshold_selection(p, f, theta):
    '''Threshold selection for Evolutionary Algorithms
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       f : ndarray
            fitness function value of individuals
       theta : float
            threshold selection parameter
       
       Returns
       -------
       ndarray
            an array of selected individuals
    '''

    sorted_indices = f.argsort()[::-1] # sort descending
    sorted_population = p[sorted_indices]

    # Threshold selection comes down to:
    mi = p.shape[0] # population size
    selected_indices = np.random.randint(0, int(theta*mi), size=mi)
    selected_population = sorted_population[selected_indices]

    return selected_population

def tournament_selection(p, f, s):
    '''Tournament selection for Evolutionary Algorithms
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       f : ndarray
            fitness function value of individuals
       s : int
            size of the tournament
       
       Returns
       -------
       ndarray
            an array of selected individuals
    '''

    sorted_indices = f.argsort()[::-1] # sort descending
    sorted_population = p[sorted_indices]
    sorted_fitness = f[sorted_indices]

    # Creating probabilities for all individuals
    ps = np.array([])
    mi = p.shape[0] # population size
    for i in range(mi):
        ps_i = 1/(mi**s)*((mi-i+2)**s - (mi-i+1)**s)
        ps = np.append(ps, ps_i)

    cp = np.cumsum(ps) # cumulative probabilities
    cp_augmented = np.concatenate( ( np.array([0]), cp ), axis=0) # falling into proper bins will resamble proportional selection probablities

    # For each new individual do tournament
    selected_population = []
    empty_out = 1
    for _ in range(mi):
        # Drawing candidates for tournament
        r = np.random.uniform(0, 1, s) # create list of random numbers for selection
        hist, _ = np.histogram(r, cp_augmented)

        # TODO: Below code can also be simplified to selecting individual for first histogram value different from 0
        tournament_individuals = []
        tournament_fitness = []
        empty_in = 1
        for j in range(mi):
            for _ in range(hist[j]):
                if empty_in:
                    tournament_individuals = sorted_population[j]
                    tournament_fitness = sorted_fitness[j]
                    empty_in = 0
                else:
                    tournament_individuals = np.vstack((tournament_individuals, sorted_population[j]))
                    tournament_fitness = np.append(tournament_fitness, sorted_fitness[j])
        winner_index = tournament_fitness.argmax()
        winner = tournament_individuals[winner_index]
        if empty_out:
            selected_population = winner
            empty_out = 0
        else:
            selected_population = np.vstack((selected_population, winner))

    return selected_population


def averaging_crossover(p, cr):
     '''Averaging crossover for Evolutionary Algorithms
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       cr : float
            crossover rate - probability that crossover occurs for individual
       
       Returns
       -------
       ndarray
            population after corssovers
     '''

     # TODO: Different methods for weight picking
     # TODO: What to do in case when drawn individuals are uneven? Maybe cr for each second individual?
     population_after_crossover = []
     for i in range(p.shape[0]):
          r_crossover = random.random()
          if r_crossover < cr:

               #select second chromosome from the population to crossover (it's possible to choose the same chromosome)
               crossed_id = random.randrange(0,p.shape[0])
               #prevent from crossing chromosome with itself
               while crossed_id == i:
                    crossed_id = random.randrange(0,p.shape[0])
               chromosome2 = p[crossed_id]
               #averaging crossover
               new_chromosome = []
               for j in range(p.shape[1]):
                    new_chromosome.append( (p[i][j]+chromosome2[j]) / 2)
               
               population_after_crossover.append(new_chromosome)
          else:
               population_after_crossover.append(p[i])
     
     return np.array([np.array(pi, dtype=object) for pi in population_after_crossover], dtype=object)

def one_point_crossover(p, cr):
     '''One point crossover for Evolutionary Algorithms

       For each individual there is a chance of one value exchange
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       cr : float
            crossover rate - probability that crossover occurs for individual
       
       Returns
       -------
       ndarray
            population after corssovers
     '''
     population_after_crossover = []
     for i in range(p.shape[0]):
          if i%2==0:
               r_crossover = random.random()
               if r_crossover < cr:
                    point = random.randrange(0,p.shape[1]-1)
                    parent1 = list(p[i])
                    parent2 = list(p[i+1])
                    
                    chromosome1 = []
                    chromosome2 = []
                    # interchanging the genes
                    for j in range(0,point):
                         chromosome1.append(parent1[j])
                         chromosome2.append(parent2[j])

                    for k in range(point, p.shape[1]):
                         chromosome1.append(parent2[k])
                         chromosome2.append(parent1[k])

                    population_after_crossover.append(chromosome1)
                    population_after_crossover.append(chromosome2)
                    
               else:
                    population_after_crossover.append(p[i])
                    population_after_crossover.append(p[i+1])

     return np.array([np.array(pi, dtype=object) for pi in population_after_crossover], dtype=object)

def even_crossover(p, cr):
     '''Even crossover for Evolutionary Algorithms

       For each individual there is an even chance of each value exchange
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       cr : float
            crossover rate - probability that crossover occurs for individual
       
       Returns
       -------
       ndarray
            population after corssovers
     '''
     population_after_crossover = []
     for i in range(p.shape[0]):
          if i%2==0:
               r_crossover = random.random()
               if r_crossover < cr:
                    exchange_vector =  [random.randint(0,1) for _ in range(p.shape[1])]
                    parent1 = list(p[i])
                    parent2 = list(p[i+1])
                    
                    chromosome1 = []
                    chromosome2 = []
                    # interchanging the genes
                    for j in range(p.shape[1]):
                         if exchange_vector[j]==1:
                              chromosome1.append(parent1[j])
                              chromosome2.append(parent2[j])
                         else:
                              chromosome1.append(parent2[j])
                              chromosome2.append(parent1[j])

                    population_after_crossover.append(chromosome1)
                    population_after_crossover.append(chromosome2)
                    
               else:
                    population_after_crossover.append(p[i])
                    population_after_crossover.append(p[i+1])

     return np.array([np.array(pi, dtype=object) for pi in population_after_crossover], dtype=object)

def mutation(p, cm,var):
     '''Mutation with normal distribution N(0,var)

       For each individual there is cm probability of mutation,
       which mutate 1/3 random selected gens in chromosome with normal distribution N(0,var)
       
    
       Parameters
       ----------
       p : ndarray
            population of individuals
       cr : float
            mutation rate - probability that mutation occurs for individual
       var : float
            variation for normal distribution
       
       Returns
       -------
       ndarray
            population after corssovers
     '''
     population_after_crossover = []
     number_mutate_gens = int(p.shape[1]/3)
     for m in range(p.shape[0]):
          r_mutation = np.random.rand(1)
          if r_mutation < cm:
               #random_value = np.random.normal(0, var, number_mutate_gens)
               nums = np.random.normal(0, var, p.shape[1])
               nums = np.random.choice([1, 0], size=p.shape[1], p=[.3, .7]) * nums
               population_after_crossover.append(p[m] + nums)
          else:
               population_after_crossover.append(p[m])
     return np.array([np.array(pi, dtype=object) for pi in population_after_crossover], dtype=object)
