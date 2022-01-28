import os

def save_algorithm_results(parameters,count_games,time):
    '''
        Save general informations about program run to file
    '''
    f = open("algorithm_results.csv", "a")
    f.write(f'{parameters},{count_games},{time}\n')
    f.close()

def save_program_results(parameters,best_fitness_list,mean_fitness_list):
    '''
        Save generation statistic about program run to file
    '''
    file_path = f"./results/program_results_{parameters}.csv"
    if os.path.isfile(file_path):
        #change values in file by adding new mean and best values to existing and update runs counter
        with open(file_path, 'r') as file :
            filedata = file.read()

        data = [] # all data from file
        string_list = list(filedata.split("\n"))
        for i in range(len(string_list)):
            if i > 0 and i < len(string_list)-1:
                splited_list =list(string_list[i].split(",") )
                integer_map = map( float, splited_list)
                integer_list = list(integer_map)
                data.append(integer_list)

        with open(file_path, 'w') as file:
            file.write('generation,best_fitness,mean_fitness\n')
            for i in range(len(best_fitness_list)):
                if i<len(data):
                    best_fitness = data[i][1] + best_fitness_list[i]
                    mean_fitness = data[i][2] + mean_fitness_list[i]
                    number = data[i][3] +1
                    file.write(f'{i},{best_fitness},{mean_fitness},{number}\n')
                else:
                    file.write(f'{i},{best_fitness_list[i]},{mean_fitness_list[i]},1\n')


    else:
        f = open(file_path, "a")
        f.write(f'generation,best_fitness,mean_fitness\n')
        for i in range(len(best_fitness_list) ): 
            f.write(f'{i},{best_fitness_list[i]},{mean_fitness_list[i]},1\n')
        f.close()
