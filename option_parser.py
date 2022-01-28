"""File with program arguments parsing methods"""
from optparse import OptionParser

def split_to_two_dim_array(value):
    string_list = list(value.split(","))
    integer_map = map(int, string_list)
    return list(integer_map)

class AppOptionParser(OptionParser):
    def __init__(self):
        super().__init__()

        self.add_option("-p", "--population", dest="population_size", default=10, type=int,
                        help="Size of population")
        self.add_option("--crossover-rate", dest="crossover_rate", default=0.75, type=float,
                        help="Crossover rate")
        self.add_option("-m", "--mutation-rate", dest="mutation_rate", default=0.75, type=float,
                        help="Mutation rate")
        self.add_option("-d", "--mutation-standard-deviation", dest="mutation_standard_deviation", default=1, type=float,
                        help="Mutation standard deviation")
        self.add_option("-n", "--hidden-neurons", dest="hidden_neurons", default="2", type=str,
                        help="Hidden neurons")
        self.add_option("-s", "--selection", dest="selection", default="proportional", type=str,
                        help="Selection type")
        self.add_option("-c", "--crossover", dest="crossover", default="averaging", type=str,
                        help="Crossover type")


    def parse_args(self, args=None, values=None):
        (options, args) = super(AppOptionParser, self).parse_args(args, values)

        if options.hidden_neurons is not None:
            options.hidden_neurons = split_to_two_dim_array(options.hidden_neurons)

        return options, args