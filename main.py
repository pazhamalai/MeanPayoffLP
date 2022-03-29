import sys
import time

import stormpy
import generateLP


def read_model(path):
    prism_program = stormpy.parse_prism_program(path)
    return stormpy.build_model(prism_program)


if __name__ == '__main__':
    start_time = time.time()
    model = read_model(sys.argv[1])
    model_read_end_time = time.time()
    print("Model Read")
    print("Number of states ", model.nr_states)
    print("Number of Transitions ", model.nr_transitions)

    generator = generateLP.LPGenerator(model, name=None)
    generator.generate()

    print("Time taken to read model ", model_read_end_time - start_time)
    print("Time taken to populate LP ", generator.get_time_taken_to_populate_lp())
    print("Time taken to solve LP ", generator.get_time_taken_to_solve_lp())
    print("Total Time Consumed", time.time() - start_time)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
