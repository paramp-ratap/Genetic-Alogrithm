
#!/usr/bin/env python3
"""
ga_cnf_solver.py

This is the GA program that imports CNF_Creator from cnf_creator.py.
Everything else (GA, evaluation, experiment scaffolding, plotting) is unchanged.
"""

import random
import time
import argparse
import math
import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# import CNF_Creator from the separated module
from CNF_Creator import *
# -------------------------
# Utilities & evaluation
# -------------------------
def evaluate_model_on_clause_list(model_bits, clauses):
    satisfied = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            var_index = abs(lit) - 1
            var_value = model_bits[var_index]
            if lit > 0 and var_value:
                sat = True
                break
            if lit < 0 and not var_value:
                sat = True
                break
        if sat:
            satisfied += 1
    return satisfied

def fitness_percent(model_bits, clauses):
    tot = len(clauses)
    if tot == 0:
        return 100.0
    sat = evaluate_model_on_clause_list(model_bits, clauses)
    return 100.0 * sat / tot

# -------------------------
# GA helpers (same as before)
# -------------------------
def random_individual(n):
    return [random.choice([False, True]) for _ in range(n)]

def tournament_selection(population, fitnesses, k=3):
    n = len(population)
    best = None
    best_f = -1.0
    for _ in range(k):
        i = random.randrange(n)
        f = fitnesses[i]
        if f > best_f:
            best_f = f
            best = population[i]
    return best.copy()

def single_point_crossover(a, b):
    n = len(a)
    if n < 2:
        return a.copy(), b.copy()
    cp = random.randrange(1, n)
    child1 = a[:cp] + b[cp:]
    child2 = b[:cp] + a[cp:]
    return child1, child2

def uniform_crossover(a, b):
    n = len(a)
    c1 = []
    c2 = []
    for i in range(n):
        if random.random() < 0.5:
            c1.append(a[i]); c2.append(b[i])
        else:
            c1.append(b[i]); c2.append(a[i])
    return c1, c2

def mutate(ind, mutation_rate):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            ind[i] = not ind[i]

def local_hill_climb(ind, clauses, max_iters=200):
    n = len(ind)
    base_fit = fitness_percent(ind, clauses)
    iters = 0
    improved = True
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            ind[i] = not ind[i]
            new_fit = fitness_percent(ind, clauses)
            if new_fit > base_fit + 1e-9:
                base_fit = new_fit
                improved = True
                break
            else:
                ind[i] = not ind[i]
    return ind

# -------------------------
# Baseline GA (unchanged)
# -------------------------
def baseline_ga(clauses, n_variables=50, pop_size=20, generations=1000,
                mutation_rate=1.0/50.0, time_limit=45.0, stagnation_limit=200):
    start_time = time.time()
    pop = [random_individual(n_variables) for _ in range(pop_size)]
    fitnesses = [fitness_percent(ind, clauses) for ind in pop]
    best_f = max(fitnesses)
    best_ind = pop[fitnesses.index(best_f)].copy()
    gens_no_improve = 0

    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        new_pop = []
        elite_idx = fitnesses.index(max(fitnesses))
        new_pop.append(pop[elite_idx].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            c1, c2 = single_point_crossover(p1, p2)
            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop
        fitnesses = [fitness_percent(ind, clauses) for ind in pop]

        cur_best_f = max(fitnesses)
        if cur_best_f > best_f + 1e-9:
            best_f = cur_best_f
            best_ind = pop[fitnesses.index(best_f)].copy()
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        if best_f >= 100.0 - 1e-9:
            break
        if gens_no_improve >= stagnation_limit:
            break

    runtime = time.time() - start_time
    return best_ind, best_f, runtime, gen+1

# -------------------------
# Improved GA (unchanged)
# -------------------------
def heuristic_seed(clause_list, n_variables):
    pos = [0]*n_variables
    neg = [0]*n_variables
    for cl in clause_list:
        for lit in cl:
            if lit > 0:
                pos[lit-1] += 1
            else:
                neg[-lit-1] += 1
    seed = []
    for i in range(n_variables):
        if pos[i] >= neg[i]:
            seed.append(True)
        else:
            seed.append(False)
    return seed

def improved_ga(clauses, n_variables=50, pop_size=60, generations=2000,
                base_mutation_rate=1.0/50.0, time_limit=45.0, stagnation_limit=300,
                keep_elite=4, local_search_elite=2):
    start_time = time.time()
    pop = []
    seed = heuristic_seed(clauses, n_variables)
    pop.append(seed.copy())
    for _ in range(3):
        s = seed.copy()
        for i in random.sample(range(n_variables), k=max(1, n_variables//20)):
            s[i] = not s[i]
        pop.append(s)
    while len(pop) < pop_size:
        pop.append(random_individual(n_variables))

    fitnesses = [fitness_percent(ind, clauses) for ind in pop]
    best_f = max(fitnesses)
    best_ind = pop[fitnesses.index(best_f)].copy()
    gens_no_improve = 0
    mutation_rate = base_mutation_rate

    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        indexed = list(enumerate(fitnesses))
        indexed.sort(key=lambda x: -x[1])
        sorted_pop = [pop[i] for i,_ in indexed]
        sorted_fit = [f for _,f in indexed]
        new_pop = [sorted_pop[i].copy() for i in range(min(keep_elite, len(sorted_pop)))]
        for i in range(min(local_search_elite, len(sorted_pop))):
            candidate = sorted_pop[i].copy()
            candidate = local_hill_climb(candidate, clauses, max_iters=200)
            new_pop.append(candidate)
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitnesses, k=3)
            p2 = tournament_selection(pop, fitnesses, k=3)
            c1, c2 = uniform_crossover(p1, p2)
            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop
        fitnesses = [fitness_percent(ind, clauses) for ind in pop]
        cur_best_f = max(fitnesses)
        if cur_best_f > best_f + 1e-9:
            best_f = cur_best_f
            best_ind = pop[fitnesses.index(best_f)].copy()
            gens_no_improve = 0
            mutation_rate = max(base_mutation_rate * 0.2, mutation_rate * 0.9)
        else:
            gens_no_improve += 1
            if gens_no_improve % 20 == 0:
                mutation_rate = min(0.2, mutation_rate * 1.4)
        if best_f >= 100.0 - 1e-9:
            break
        if gens_no_improve >= stagnation_limit:
            break

    runtime = time.time() - start_time
    return best_ind, best_f, runtime, gen+1

# -------------------------
# Experiment scaffolding & plotting (unchanged)
# -------------------------
def run_single_trial(clauses, n_variables, mode='baseline'):
    if mode == 'baseline':
        best_ind, best_f, runtime, gens = baseline_ga(clauses, n_variables=n_variables)
    else:
        best_ind, best_f, runtime, gens = improved_ga(clauses, n_variables=n_variables)
    return best_ind, best_f, runtime, gens

def experiment_for_m_list(ms, repeats=10, n_variables=50, mode='improved'):
    avg_fitness = {}
    avg_runtime = {}
    cnf = CNF_Creator(n_variables)
    for m in ms:
        fitnesses = []
        runtimes = []
        print(f"Starting experiments for m={m}, repeats={repeats}, mode={mode}")
        for r in range(repeats):
            clauses = cnf.CreateRandomSentence(m)
            print(len(clauses))
            _, best_f, runtime, _ = run_single_trial(clauses, n_variables, mode=mode)
            fitnesses.append(best_f)
            runtimes.append(runtime)
            print(f"  repeat {r+1}/{repeats}: best_f={best_f:.3f} runtime={runtime:.2f}s")
        avg_fitness[m] = sum(fitnesses)/len(fitnesses)
        avg_runtime[m] = sum(runtimes)/len(runtimes)
        print(f"  -> avg_fitness={avg_fitness[m]:.3f}, avg_runtime={avg_runtime[m]:.3f}s")
    return avg_fitness, avg_runtime

def plot_results(avg_fitness, avg_runtime, out_prefix='results'):
    ms = sorted(avg_fitness.keys())
    fitness_vals = [avg_fitness[m] for m in ms]
    runtime_vals = [avg_runtime[m] for m in ms]

    plt.figure(figsize=(8,5))
    plt.plot(ms, fitness_vals, marker='o')
    plt.xlabel('Number of clauses m')
    plt.ylabel('Average best fitness (%)')
    plt.title('Average best fitness vs m')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_fitness.png')
    print(f"Saved {out_prefix}_fitness.png")

    plt.figure(figsize=(8,5))
    plt.plot(ms, runtime_vals, marker='o')
    plt.xlabel('Number of clauses m')
    plt.ylabel('Average runtime (s)')
    plt.title('Average runtime vs m')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_runtime.png')
    print(f"Saved {out_prefix}_runtime.png")

# -------------------------
# CLI entry
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','improved','experiment'], default='experiment')
    parser.add_argument('--ms', nargs='+', type=int, default=[100,120,140,160,180,200,220,240,260,280,300])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()

    random.seed(42)
    print("Roll No : 2022A7PS1124G")
    print("")
    if args.mode in ('baseline','improved'):
        m = 300
        cnf = CNF_Creator(args.n)
        clauses = cnf.CreateRandomSentence(m)
        print(len(clauses))
        if args.mode == 'baseline':
            best, bf, rt, gens = baseline_ga(clauses, n_variables=args.n)
        else:
            best, bf, rt, gens = improved_ga(clauses, n_variables=args.n)
        print(f"Mode {args.mode}: best fitness {bf:.4f}%, runtime {rt:.2f}s, gens {gens}")
    else:
        ms = args.ms
        avg_fitness, avg_runtime = experiment_for_m_list(ms, repeats=args.repeats, n_variables=args.n, mode='improved')
        plot_results(avg_fitness, avg_runtime, out_prefix=f'ga_{args.mode}')

if __name__=='__main__':
    main()

