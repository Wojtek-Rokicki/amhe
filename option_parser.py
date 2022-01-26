from optparse import OptionParser


def split_to_two_dim_array(value):
    string_list = list(value.split(","))
    integer_map = map(int, string_list)
    return list(integer_map)

class AppOptionParser(OptionParser):
    def __init__(self):
        super().__init__()
        # self.add_option("-g", "--generations", dest="generations", default=10000, type=int,
        #                 help="Number of generations")
        self.add_option("-p", "--population", dest="population_size", default=10, type=int,
                        help="Size of population")
        self.add_option("-c", "--crossover_rate", dest="crossover_rate", default=0.5, type=float,
                        help="Crossover rate")
        self.add_option("-m", "--mutation_rate", dest="mutation_rate", default=0.5, type=float,
                        help="Mutation rate")
        self.add_option("-v", "--mutation_variation", dest="mutation_variation", default=1, type=float,
                        help="Mutation variation")
        self.add_option("-n", "--hidden_neurons", dest="hidden_neurons", default="2", type=str,
                        help="Hidden neurons")

        self.add_option("-s", "--selection", dest="selection", default="proportional", type=str,
                        help="Selection type")
        self.add_option("-k", "--crossover", dest="crossover", default="even", type=str,
                        help="Crossover type")


    def parse_args(self, args=None, values=None):
        (options, args) = super(AppOptionParser, self).parse_args(args, values)

        if options.hidden_neurons is not None:
            options.hidden_neurons = split_to_two_dim_array(options.hidden_neurons)
            # print(type(options.hidden_neurons))

        return options, args