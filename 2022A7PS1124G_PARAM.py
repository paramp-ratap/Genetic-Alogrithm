import random
import time
import sys
from CNF_Creator import *

def run_evolution(cnf_clauses_list,
                  number_of_variables=50,
                  population_size=60,
                  elitism_count=4,
                  local_search_count=2,
                  base_mutation_rate=1.0 / 50.0,
                  allowed_time_seconds=44.0,
                  maximum_generations=2000,
                  stagnation_limit=300,
                  random_seed_value=None):

    # Deterministic seed for debugging unless user supplied one
    random.seed(42)

    start_time = time.time()  # start time for the evolutionary run

    def calc_percentage(numerator, denominator):
        return 100.0 * numerator / denominator if denominator else 100.0  # avoid div0

    def compute_fitness_percentage(valuation_bitlist):
        return calc_percentage(count_satisfied_clauses(valuation_bitlist),
                               len(cnf_clauses_list))  # percent satisfied

    def generate_random_bitstring(n_bits):
        return [random.choice([False, True]) for _ in range(n_bits)]  # random init

    def count_satisfied_clauses(valuation_bitlist):
        satisfied_count = 0
        for clause in cnf_clauses_list:
            for literal in clause:
                val = valuation_bitlist[abs(literal) - 1]
                if (literal > 0 and val) or (literal < 0 and not val):
                    satisfied_count += 1
                    break
        return satisfied_count  # how many clauses satisfied

    def tournament_selection(population_list, fitness_values_list, tournament_k=3):
        selected = None
        best_score_seen = -1.0
        for _ in range(tournament_k):
            idx = random.randrange(len(population_list))
            score = fitness_values_list[idx]
            if score > best_score_seen:
                best_score_seen = score
                selected = population_list[idx]
        return selected.copy()  # return a copy of the selected individual

    def local_hill_climb(individual_bitlist, max_local_steps=200):
        current_score = compute_fitness_percentage(individual_bitlist)
        n = len(individual_bitlist)
        for _ in range(max_local_steps):
            found_improvement = False
            explore_order = list(range(n))
            random.shuffle(explore_order)  # randomize local flip order
            for index_to_flip in explore_order:
                individual_bitlist[index_to_flip] = not individual_bitlist[index_to_flip]
                new_score = compute_fitness_percentage(individual_bitlist)
                if new_score > current_score:
                    current_score = new_score
                    found_improvement = True
                    break
                # revert if no improvement
                individual_bitlist[index_to_flip] = not individual_bitlist[index_to_flip]
            if not found_improvement:
                break
        return individual_bitlist  # return potentially improved individual

    def uniform_crossover(parent_a, parent_b):
        n = len(parent_a)
        child_x = [parent_a[i] if random.random() < 0.5 else parent_b[i] for i in range(n)]
        child_y = [parent_b[i] if random.random() < 0.5 else parent_a[i] for i in range(n)]
        return child_x, child_y  # two children

    def mutate_flip_bits(individual_bitlist, mutation_probability):
        for idx in range(len(individual_bitlist)):
            if random.random() < mutation_probability:
                individual_bitlist[idx] = not individual_bitlist[idx]  # flip bit

    def majority_seed_initialization(n_variables):
        positive_counts = [0] * n_variables
        negative_counts = [0] * n_variables
        for clause in cnf_clauses_list:
            for literal in clause:
                if literal > 0:
                    positive_counts[literal - 1] += 1
                else:
                    negative_counts[-literal - 1] += 1
        return [positive_counts[i] >= negative_counts[i] for i in range(n_variables)]

    # --- initialize population ---

    population = []
    initial_seed_individual = (majority_seed_initialization(number_of_variables)
                               if cnf_clauses_list else generate_random_bitstring(number_of_variables))
    population.append(initial_seed_individual.copy())  # seeded individual

    for _ in range(3):
        noisy_variant = initial_seed_individual.copy()
        for j in random.sample(range(number_of_variables),
                               max(1, number_of_variables // 20)):
            noisy_variant[j] = not noisy_variant[j]
        population.append(noisy_variant)  # small noisy variants

    while len(population) < population_size:
        population.append(generate_random_bitstring(number_of_variables))  # fill rest randomly

    fitness_values = [compute_fitness_percentage(ind) for ind in population]
    best_fitness_value = max(fitness_values)
    best_solution_model = population[fitness_values.index(best_fitness_value)].copy()
    stagnation_counter = 0
    mutation_rate = base_mutation_rate  # current mutation rate

    last_generation_index = 0
    for last_generation_index in range(1, maximum_generations + 1):
        # wall-clock stop
        if time.time() - start_time > allowed_time_seconds:
            break

        sorted_indices_by_fitness = sorted(range(len(fitness_values)), key=lambda i: -fitness_values[i])
        sorted_population = [population[i] for i in sorted_indices_by_fitness]

        new_population = [sorted_population[i].copy() for i in range(min(elitism_count, len(sorted_population)))]

        # local search on top individuals
        for i_local in range(min(local_search_count, len(sorted_population))):
            improved_individual = local_hill_climb(sorted_population[i_local].copy())
            new_population.append(improved_individual)

        # fill rest of the population with offspring
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child1, child2 = uniform_crossover(parent1, parent2)
            mutate_flip_bits(child1, mutation_rate)
            mutate_flip_bits(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        fitness_values = [compute_fitness_percentage(ind) for ind in population]

        current_generation_best = max(fitness_values)
        if current_generation_best > best_fitness_value:
            best_fitness_value = current_generation_best
            best_solution_model = population[fitness_values.index(best_fitness_value)].copy()
            stagnation_counter = 0
            mutation_rate = max(base_mutation_rate * 0.2, mutation_rate * 0.9)  # reduce mutation on success
        else:
            stagnation_counter += 1
            if stagnation_counter % 20 == 0:
                mutation_rate = min(0.25, mutation_rate * 1.4)  # increase mutation occasionally

        if best_fitness_value >= 100.0 or stagnation_counter >= stagnation_limit:
            break  # solved or stagnation stop

    elapsed_seconds = time.time() - start_time
    return best_solution_model, best_fitness_value, elapsed_seconds, last_generation_index  # result tuple


def main_run():
    TOTAL_ALLOWED = 44.0
    main_start_time = time.time()

    cnf_creator = CNF_Creator(50)
    cnf_clauses_list = cnf_creator.ReadCNFfromCSVfile()

    after_read_time = time.time()
    read_duration_seconds = after_read_time - main_start_time
    time_left_for_ga = TOTAL_ALLOWED - read_duration_seconds
    if time_left_for_ga <= 0:
        print("Roll No : 2022A7PS1124G")
        print(f"Number of clauses in CSV file : {len(cnf_clauses_list)}")
        print("Time limit reached before GA could start.")
        sys.exit(1)  # no time left

    model_solution, model_fitness_value, ga_elapsed_seconds, generations_ran = run_evolution(
        cnf_clauses_list,
        number_of_variables=50,
        allowed_time_seconds=time_left_for_ga
    )

    signed_model_representation = [(i + 1) if model_solution[i] else -(i + 1)
                                   for i in range(len(model_solution))]
    total_elapsed_time = time.time() - main_start_time
    print("Roll No : 2022A7PS1124G")
    print(f"Number of clauses in CSV file : {len(cnf_clauses_list)}")
    print(f"Best model : [{', '.join(str(x) for x in signed_model_representation)}]")
    print(f"Fitness value of best model : {int(round(model_fitness_value))}%")
    print(f"Time taken : {total_elapsed_time:.2f} seconds")  # overall time


if __name__ == "__main__":
    main_run()
