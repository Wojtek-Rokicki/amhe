''' Rile with functions making plots
'''
import matplotlib.pyplot as plt
import pandas as pd

def get_population_size(name):
    string_list = list(name.split("_"))
    return int(string_list[0])

def get_population_selection_crossover(name):
    string_list = list(name.split("_"))
    return string_list[5]+"_"+string_list[6]

def get_crossover_mutation_rate(name):
    string_list = list(name.split("_"))
    return string_list[1]+"_"+string_list[2]

def get_network_param(name):
    string_list = list(name.split("_"))
    print(string_list)
    return string_list[4]

def plot_algorithm_results(start_row, end_row, file_name, save_file_name,function_params, x_axis_label,title):

    df = pd.read_csv (file_name,usecols= ['params','games','time'])

    df_sel_cro = df[start_row:end_row].copy()
    df_sel_cro['params'] = df_sel_cro['params'].apply(function_params)
    df_sel_cro_time = df_sel_cro.groupby('params', as_index=False)['time'].mean()
    print(df_sel_cro_time)
    df_sel_cro_games = df_sel_cro.groupby('params', as_index=False)['games'].mean()
    print(df_sel_cro_games)

    f3 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df_sel_cro_time['params'].tolist()
    y=df_sel_cro_time['time'].tolist()
    plt.scatter(x, y)
    plt.xticks(rotation=90)
    plt.title(f'Czas działania algorytmu dla {title}')
    plt.ylabel("czas [s]")
    plt.xlabel(x_axis_label)
    f3.savefig(f'./plots/{save_file_name}_times.png')

    f4 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df_sel_cro_games['params'].tolist()
    y=df_sel_cro_games['games'].tolist()
    plt.scatter(x, y)
    plt.xticks(rotation=90)
    plt.title(f'Liczba rozegranych gier algorytmu dla {title}')
    plt.ylabel("liczba rozegranych gier")
    plt.xlabel(x_axis_label)
    f4.savefig(f'./plots/{save_file_name}_games.png')

def plot_program_results(file):
    file = './results/program_results_10_0.5_0.5_1_[2]_proportional_one_point.csv'
    df = pd.read_csv (file,usecols= ['generation','best_fitness','mean_fitness','number'])
    df['average_best'] = df['best_fitness']/df[ 'number']
    df['average_mean'] = df['mean_fitness']/df[ 'number']
    print(df)

    plt.subplots_adjust(bottom=0.45)
    x= df['generation'].tolist()
    y=df['average_best'].tolist()
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.title('Najlepszy osobnik w populacji')
    plt.ylabel("wynik")
    plt.xlabel("generacja")
    plt.savefig('./plots//try_best.png')

    plt.subplots_adjust(bottom=0.45)
    x= df['generation'].tolist()
    y=df['average_mean'].tolist()
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.title('Najlepszy osobnik w populacji')
    plt.ylabel("wynik")
    plt.xlabel("generacja")
    plt.savefig('./plots/try_mean.png')

def plot_dqn_results():
    file = 'dqn.csv'
    df = pd.read_csv (file,usecols= ['game_no','exploration_rate','score','min_score','average_score','max_score'])

    plt.subplots_adjust(bottom=0.45)
    x= df['game_no'].tolist()
    y1=df['score'].tolist()
    plt.plot(x, y1)
    plt.xticks(rotation=90)
    plt.title('Kolejne symulacje algorytmu DQN')
    plt.ylabel("wynik")
    plt.xlabel("generacja")

    plt.subplots_adjust(bottom=0.45)
    y2=df['average_score'].tolist()
    plt.plot(x, y2)
    plt.legend(["liczba nagród", "średnia liczba nagród"], loc ="upper left")
    plt.savefig('./plots/dqn_analysis.png')


if __name__ == "__main__":
    plot_algorithm_results()
    # plot_algorithm_results(1,18,'./results/algorithm_results_copy', 'population_changing', get_population_size, 'Liczba osobników w populacji', 'różnej liczebności populacji')
    # plot_algorithm_results(19,89,'./results/algorithm_results_copy', 'selection_crossover_changing', get_population_selection_crossover, 'Typ selekcji i krzyżowania', 'różnych selekcji i krzyżowania')
    plot_dqn_results()
    plot_algorithm_results(1, 42, './results/parameters_experiments_results.csv', 'parameters_experiments_results',get_crossover_mutation_rate,'prawdopodobieństwo krzyżowania i mutacji','różnych ustawień parametrów')
    plot_algorithm_results(1, 50, './results/population_size_experiments_10-100.csv', 'population_size_experiments_10-100_results',get_population_size,'rozmiar populacji','różnych rozmiarów populacji')
    plot_algorithm_results(1, 50, './results/population_size_experiments_100-1000.csv', 'population_size_experiments_100-1000_results',get_population_size,'rozmiar populacji','różnych rozmiarów populacji')
    plot_algorithm_results(1, 60, './results/network_experiments.csv', 'network_experiments_results',get_network_param,'liczby neuronów w warstwach uruchamianej sieci','różnych sieci neuronowych')
