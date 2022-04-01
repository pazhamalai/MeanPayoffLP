import sys
import time

import stormpy
from LPSolver import LPSolver


def read_model(file_name):
    prism_program = stormpy.parse_prism_program("./models/" + file_name)
    return stormpy.build_model(prism_program)


def solve_mean_payoff(file_name):
    start_time = time.time()
    model = read_model(file_name)
    model_read_end_time = time.time()
    # print("Model Read")
    # print("Number of states ", model.nr_states)
    # print("Number of Transitions ", model.nr_transitions)

    lp_solver = LPSolver(model, name=None)
    result = lp_solver.solve_mean_payoff()

    # print("Time taken to read model ", model_read_end_time - start_time)
    # print("Time taken to populate LP ", lp_solver.get_time_taken_to_populate_lp())
    # print("Time taken to solve LP ", lp_solver.get_time_taken_to_solve_lp())
    # print("Total Time Consumed", time.time() - start_time)
    return result


if __name__ == '__main__':
    solve_mean_payoff(sys.argv[1])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
