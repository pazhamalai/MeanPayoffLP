import sys
import stormpy
import generateLP


def read_model(path):
    prism_program = stormpy.parse_prism_program(path)
    return stormpy.build_model(prism_program)


if __name__ == '__main__':
    model = read_model(sys.argv[1])
    generator = generateLP.LPGenerator(model, name=None)
    generator.generate()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
