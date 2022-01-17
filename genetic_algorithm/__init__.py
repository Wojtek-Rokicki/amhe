import numpy as np

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
    return

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

    return

def even_crossover():
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

    return