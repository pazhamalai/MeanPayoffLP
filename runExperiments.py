# For each file print the mean-payoff, time it took to compute the mean-payoff
# Run 10 iterations for each model.
import os
import time
import shutil
import meanPayoffSolver as mps

directory = 'models'
output_directory = 'result'
n_experiments = 2

models = ["blackjack.prism", "consensus.2.prism", "counter.prism", "cs_nfail3.prism", "ij.3.prism",
          "ij.10.prism", "investor.prism", "pacman.prism", "recycling.prism", "sensors.prism", "virus.prism",
          "wlan.0.prism"]

other_benchmarks = ["ns5.pm", "ns6.pm", "ns15.pm", "ns16.pm", "ns18.pm", "ns100.pm",
                    "ns101.pm", "soft6.pm", "soft10.pm", "soft15.pm"]


def write_result(file_name, mean_payoffs, time_taken):
    aggregate_mean_payoff = 0
    aggregate_time_taken = 0
    for i in range(n_experiments):
        aggregate_mean_payoff += mean_payoffs[i]
        aggregate_time_taken += time_taken[i]
    average_result = aggregate_mean_payoff / n_experiments
    average_time_taken = aggregate_time_taken / n_experiments

    with open('./result/' + file_name + '.txt', 'w') as result_file:
        for i in range(n_experiments):
            result_file.write(str(mean_payoffs[i]))
            result_file.write("\n")
            result_file.write(str(time_taken[i]))
            result_file.write("\n")

        result_file.write("Average Mean-payoff: " + str(average_result))
        result_file.write("\n")
        result_file.write("Average time taken: " + str(average_time_taken))


def run_experiments_for_file(filename):
    global n_experiments
    mean_payoffs = []
    times = []
    for _ in range(n_experiments):
        start_time = time.time()
        mean_payoffs.append(mps.solve_mean_payoff(filename))
        times.append(time.time() - start_time)

    write_result(filename, mean_payoffs, times)


def remove_old_results():
    is_dir_exists = os.path.isdir(output_directory)
    if is_dir_exists:
        shutil.rmtree(output_directory)


def create_new_output_directory():
    os.mkdir(output_directory)


def init_output_directory():
    remove_old_results()
    create_new_output_directory()


init_output_directory()
for filename in models:
    run_experiments_for_file(filename)

for filename in other_benchmarks:
    run_experiments_for_file(filename)
