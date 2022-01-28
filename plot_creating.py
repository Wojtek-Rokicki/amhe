import matplotlib.pyplot as plt
import pandas as pd

def get_population_size(name):
    string_list = list(name.split("_"))
    return string_list[0]

def get_population_selection_crossover(name):
    string_list = list(name.split("_"))
    return string_list[5]+"_"+string_list[6]

def plot_algorithm_results():
    file = 'algorithm_results_copy.csv'

    df = pd.read_csv (file,usecols= ['params','games','time'])

    #get first rows with information about algorithm run
    df_pop = df.head(18).copy()
    df_pop['params'] = df_pop['params'].apply(get_population_size)
    x= df_pop['params'].tolist()
    y=df_pop['time'].tolist()

    f1 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    plt.scatter(x, y)
    plt.xticks(rotation=45)
    plt.title('Czas działania algorytmu dla różnej liczebności populacji')
    plt.ylabel("czas [s]")
    plt.xlabel("Liczba osobników w populacji")
    f1.savefig('population_changing_time.png')

    f2 = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    y=df_pop['games'].tolist()
    plt.scatter(x, y)
    plt.xticks(rotation=45)
    plt.title('Liczba rozegranych gier algorytmu dla różnej liczebności populacji')
    plt.ylabel("liczba rozegranych gier")
    plt.xlabel("Liczba osobników w populacji")
    f2.savefig('population_changing_games.png')

    df_sel_cro = df[19:89].copy()
    df_sel_cro['params'] = df_sel_cro['params'].apply(get_population_selection_crossover)
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
    plt.title('Czas działania algorytmu dla różnych selekcji i krzyżowania')
    plt.ylabel("czas [s]")
    plt.xlabel("Typ selekcji i krzyżowania")
    f3.savefig('selection_crossover_changing_times.png')

    f4 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df_sel_cro_games['params'].tolist()
    y=df_sel_cro_games['games'].tolist()
    plt.scatter(x, y)
    plt.xticks(rotation=90)
    plt.title('Liczba rozegranych gier algorytmu dla różnych selekcji i krzyżowania')
    plt.ylabel("liczba rozegranych gier")
    plt.xlabel("Typ selekcji i krzyżowania")
    f4.savefig('selection_crossover_changing_games.png')

def plot_program_results():
    file = './results/program_results_10_0.5_0.5_1_[2]_proportional_one_point.csv'
    df = pd.read_csv (file,usecols= ['generation','best_fitness','mean_fitness','number'])
    df['average_best'] = df['best_fitness']/df[ 'number']
    df['average_mean'] = df['mean_fitness']/df[ 'number']
    print(df)

    # f1 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df['generation'].tolist()
    y=df['average_best'].tolist()
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.title('Najlepszy osobnik w populacji')
    plt.ylabel("wynik")
    plt.xlabel("generacja")
    plt.savefig('try_best.png')

    # f2 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df['generation'].tolist()
    y=df['average_mean'].tolist()
    plt.plot(x, y)
    plt.xticks(rotation=90)
    plt.title('Najlepszy osobnik w populacji')
    plt.ylabel("wynik")
    plt.xlabel("generacja")
    plt.savefig('try_mean.png')

def plot_dqn_results():
    file = 'dqn.csv'
    df = pd.read_csv (file,usecols= ['game_no','exploration_rate','score','min_score','average_score','max_score'])
    # df['average_best'] = df['best_fitness']/df[ 'number']
    # df['average_mean'] = df['mean_fitness']/df[ 'number']
    # print(df)

    # f1 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    x= df['game_no'].tolist()
    y1=df['score'].tolist()
    plt.plot(x, y1)
    plt.xticks(rotation=90)
    plt.title('Kolejne symulacje algorytmu DQN')
    plt.ylabel("wynik")
    plt.xlabel("generacja")
    # plt.savefig('try_best.png')

    # f2 = plt.figure()
    plt.subplots_adjust(bottom=0.45)
    y2=df['average_score'].tolist()
    plt.plot(x, y2)
    plt.legend(["liczba nagród", "średnia liczba nagród"], loc ="upper left")
    plt.savefig('dqn_analysis.png')


if __name__ == "__main__":
    # plot_algorithm_results()
    # plot_program_results()
    plot_dqn_results()