#!/usr/bin/env python3

import pandas as pd
import xarray as xr
import numpy as np
import copy
import math
import random
import warnings
import matplotlib.pyplot as plt
import os
import time

import calliope
from calliope.exceptions import ModelWarning

calliope.set_log_verbosity(verbosity='critical', include_solver_output=False, capture_warnings=False)

# Suppress the specific ModelWarning from Calliope
warnings.filterwarnings("ignore", category=ModelWarning)

from deap import base
from deap import creator
from deap import tools

#population and generation
NUM_SUBPOPS = 2
SUBPOP_SIZE = 4 
GENERATIONS = 5

#crossover
CROSSR = 0.5 

#mutation 
INDMUT = 0.3 
INDMUT2 = 0.1

#mutbound eta value
ETAV = 1 
ETAV2 = 3
ETAV3 = 7

#result location
results_dir = "results_1"
os.makedirs(results_dir, exist_ok=True)

# value changes
value_change_1 = 30  
value_change_2 = 70 

#resolution changes
resolution_change_1 = 80 #3.5D
resolution_change_2 = 95 #6H


#create the model with a resolution of 14 days
model_14D = calliope.Model("C:/Users/Jacob/Desktop/PythonProjects/GAMGA-Calliope v3.9/GAMGA_model/model_14D.yaml")

model = model_14D

model.run()

df_total_cost = model.results.cost.to_series().dropna()
total_cost_optimal = df_total_cost.loc[~df_total_cost.index.map(str).str.contains('co2_emissions')].sum()

print(total_cost_optimal)

energy_cap_df = model.results.energy_cap.to_pandas()
filtered_energy_cap_df = energy_cap_df[~energy_cap_df.index.str.contains("demand|transmission")]

print(filtered_energy_cap_df)


#create the model with a resolution of 3.5 days
model_3_5D = calliope.Model('C:/Users/Jacob/Desktop/PythonProjects/GAMGA-Calliope v3.9/GAMGA_model/model_3.5D.yaml')

model_3_5D.run()

df_total_cost_3_5D = model_3_5D.results.cost.to_series().dropna()
total_cost_optimal_3_5D = df_total_cost_3_5D.loc[~df_total_cost_3_5D.index.map(str).str.contains('co2_emissions')].sum()

print(total_cost_optimal_3_5D)

energy_cap_df_3_5D = model_3_5D.results.energy_cap.to_pandas()
filtered_energy_cap_df_3_5D = energy_cap_df_3_5D[~energy_cap_df_3_5D.index.str.contains("demand|transmission")]

print(filtered_energy_cap_df_3_5D)

#create the model with a resolution of 6 hours
model_6H = calliope.Model('C:/Users/Jacob/Desktop/PythonProjects/GAMGA-Calliope v3.9/GAMGA_model/model_6H.yaml')

model_6H.run()

df_total_cost_6H = model_6H.results.cost.to_series().dropna()
total_cost_optimal_6H = df_total_cost_6H.loc[~df_total_cost_6H.index.map(str).str.contains('co2_emissions')].sum()

print(total_cost_optimal_6H)

energy_cap_df_6H = model_6H.results.energy_cap.to_pandas()
filtered_energy_cap_df_6H = energy_cap_df_6H[~energy_cap_df_6H.index.str.contains("demand|transmission")]

print(filtered_energy_cap_df_6H)


# feasibility values
optimal_value = total_cost_optimal_6H
max_slack = 0.2
unmetdemandvariable = 30000

#find Max capacity to use for inf values
#initial_capacities = filtered_energy_cap_df.values 

initial_capacities_6H = filtered_energy_cap_df_6H.values 
max_cap_value = max(filtered_energy_cap_df)
print(max_cap_value)
print(initial_capacities_6H)


updates = [
    {'tech': tech, 'loc': loc}
    for loc_tech in filtered_energy_cap_df.index
    for loc, tech in [loc_tech.split("::")]  # Split index by '::' to separate loc and tech
]
print(updates)


input_params = model.backend.access_model_inputs()

# Access energy_cap_max and energy_cap_min
energy_cap_max = input_params['energy_cap_max']
energy_cap_min = input_params['energy_cap_min']

# Convert to DataFrame for filtering
energy_cap_max_df = energy_cap_max.to_dataframe()
energy_cap_min_df = energy_cap_min.to_dataframe()

# Filter out rows with 'demand' or 'free' in the index
energy_cap_max_filtered = energy_cap_max_df[~energy_cap_max_df.index.get_level_values('loc_techs').str.contains("demand|transmission")]
energy_cap_min_filtered = energy_cap_min_df[~energy_cap_min_df.index.get_level_values('loc_techs').str.contains("demand|transmission")]


# Create a dictionary of loc_tech to [min, max] bounds
low_up_mapping = {
    loc_tech: [
        energy_cap_min_filtered.loc[loc_tech, 'energy_cap_min'],
        energy_cap_max_filtered.loc[loc_tech, 'energy_cap_max']
    ]
    for loc_tech in energy_cap_max_filtered.index
}

# Generate low_up_bound based on the order in updates
updates_order = [f"{update['loc']}::{update['tech']}" for update in updates]

low_up_bound = [
    low_up_mapping[loc_tech] for loc_tech in updates_order
]

# Check for 'inf' in upper bounds and adjust if needed
max_cap_value = 100000  # Replace with a reasonable maximum if applicable
for i, (low, up) in enumerate(low_up_bound):
    if up == float('inf'):
        print(f"Technology {updates_order[i]} has 'inf' as the upper bound.")
        low_up_bound[i][1] = max_cap_value
    else:
        print(f"Technology {updates_order[i]} has a finite upper bound: {up}")

print("Updated low_up_bound:", low_up_bound)


def update_energy_cap_max_for_individual(model, updates, individual_values):

    # Ensure the length of updates matches the individual's values
    if len(updates) != len(individual_values):
        raise ValueError("Length of updates and individual values must match.")
    
    # Update the model with the individual's capacity values
    for update, new_cap in zip(updates, individual_values):
        tech = update['tech']
        loc = update['loc']
        
        # Construct the location::technology key and update the model
        loc_tech_key = f"{loc}::{tech}"
        model.backend.update_param('energy_cap_max', {loc_tech_key: new_cap})
        model.backend.update_param('energy_cap_min', {loc_tech_key: new_cap})
    
    # Run the model for this individual
    try:
        rerun_model = model.backend.rerun()  # Rerun to capture updated backend parameters

        # Calculate the total cost, excluding emission costs
        cost_op = rerun_model.results.cost.to_series().dropna()
        initial_cost = round(cost_op.loc[~cost_op.index.map(str).str.contains('co2_emissions')].sum(), 2)

        unmet = rerun_model.results.unmet_demand.to_series().dropna()
        unmet_demand = round(unmet.sum() * unmetdemandvariable, 2) #300 is the penalty for unmet demand

        total_cost = initial_cost + unmet_demand
    
    except Exception as e:
        # If solving fails, set total cost to NaN and print a warning
        total_cost = float('inf')
        print("Warning: Model could not be solved for the individual. Assigning cost as infinite.")
    
    return total_cost

def slack_feasibility(individual):
    cost = update_energy_cap_max_for_individual(model, updates, individual)
    individual.cost = cost  # Attach cost attribute to individual
    slack_distance = (cost - optimal_value) / optimal_value

    # Update feasibility condition based on the new criteria
    feasible = slack_distance <= max_slack #and cost >= optimal_value

    print(f"Slack feasibility for individual: {feasible}, Cost: {individual.cost}")
    
    return feasible 

def centroidSP(subpop):
    centroids = []

    # Iterate over each subpopulation and calculate the centroid
    for sub in subpop.values():
        if not isinstance(sub, list) or not all(isinstance(individual, list) for individual in sub):
            raise TypeError("Each subpopulation must be a list of lists (individuals).")
        
        num_solutions = len(sub)  # Number of solutions in the current subpopulation
        num_variables = len(sub[0])  # Number of decision variables
        
        # Calculate the centroid for each decision variable
        centroid = [sum(solution[i] for solution in sub) / num_solutions for i in range(num_variables)]
        centroids.append(centroid)  # Append each centroid to the main list in the required format
    
    return centroids

def fitness(subpop, centroids):
    distances = []
    minimal_distances = []
    fitness_SP = {}

    # Step 1: Calculate Distances per Variable for each individual
    for q, (subpop_index, subpopulation) in enumerate(subpop.items()):
        subpopulation_distances = []
        
        for individual in subpopulation:
            individual_variable_distances = []
            
            for p, centroid in enumerate(centroids):
                if p != q:  # Skip the centroid of the same subpopulation
                    variable_distances = [abs(individual[i] - centroid[i]) for i in range(len(individual))]
                    individual_variable_distances.append(variable_distances)
            
            subpopulation_distances.append(individual_variable_distances)
        
        distances.append(subpopulation_distances)

    # Step 2: Calculate Minimal Distances per Variable
    for subpopulation_distances in distances:
        subpopulation_minimal = []
        
        for individual_distances in subpopulation_distances:
            min_distance_per_variable = [min(distance[i] for distance in individual_distances) for i in range(len(individual_distances[0]))]
            subpopulation_minimal.append(min_distance_per_variable)
        
        minimal_distances.append(subpopulation_minimal)

    # Step 3: Calculate Fitness SP for each individual
    for sp_index, subpopulation in enumerate(minimal_distances, start=1):
        fitness_values = [(min(individual),) for individual in subpopulation]
        fitness_SP[sp_index] = fitness_values

    return fitness_SP

def custom_tournament(subpopulation, k, tournsize=2):
    selected = []
    zero_fitness_count = 0  # Counter for individuals with fitness (0,)

    while len(selected) < k:
        # Randomly select `tournsize` individuals for the tournament
        tournament = random.sample(subpopulation, tournsize)

        # Check if all individuals in the tournament have a fitness of (0,)
        if all(ind.fitness.values == (0,) for ind in tournament):
            if zero_fitness_count < 2:
                # Select the individual with the lowest cost if all fitness values are (0,)
                best = min(tournament, key=lambda ind: ind.cost)
                selected.append(best)
                zero_fitness_count += 1
            else:
                # Select a random feasible individual if we've reached the max count of (0,) fitness values
                feasible_individuals = [ind for ind in subpopulation if ind.fitness.values != (0,)]
                if feasible_individuals:
                    best = random.choice(feasible_individuals)
                    selected.append(best)
                else:
                    # If no feasible individuals are available, fallback to random selection to avoid empty selection
                    best = random.choice(subpopulation)
                    selected.append(best)
        else:
            # Select based on fitness if there are feasible individuals in the tournament
            best = max(tournament, key=lambda ind: ind.fitness.values[0])
            selected.append(best)

    return selected

def generate_individual():
    # Generate a new individual with capacities within the defined bounds
    adjusted_individual = [
        max(low, min(up, cap + random.choice([-1, 0, 1])))
        for cap, (low, up) in zip(initial_capacities_6H, low_up_bound)]
    return adjusted_individual

# DEAP creator setup
creator.create("FitnessMaxDist", base.Fitness, weights=(1.0,))  # Fitness to maximize distinctiveness
creator.create("IndividualSP", list, fitness=creator.FitnessMaxDist, cost=0)  # Individual structure in DEAP

# DEAP toolbox setup
toolbox = base.Toolbox()

# Register the individual and subpopulation initializers
toolbox.register("individualSP", tools.initIterate, creator.IndividualSP, generate_individual)
toolbox.register("subpopulationSP", tools.initRepeat, list, toolbox.individualSP)

#register the operators
toolbox.register("mate", tools.cxUniform)
toolbox.register("elitism", tools.selBest, fit_attr="fitness.values")
toolbox.register("tournament", custom_tournament)
toolbox.register("mutbound", tools.mutPolynomialBounded)


for run in range(1, 6):  # 5 repetitions
    print(f"--- Starting Full Run {run} ---")

    start_time = time.time()

    # Generate subpopulations with multiple individuals
    subpops_unaltered = [toolbox.subpopulationSP(n=SUBPOP_SIZE) for _ in range(NUM_SUBPOPS)]

    subpops_SP = {}

    for p in range(NUM_SUBPOPS):
        subpops_SP[p+1] = subpops_unaltered[p]

    #calculate centroids and fitness
    centroids = centroidSP(subpops_SP)
    fitness_populations = fitness(subpops_SP, centroids)

    # Combine the fitness values with each individual
    for i, subpopulation in subpops_SP.items():
        for individual, fit in zip(subpopulation, fitness_populations[i]):
            individual.fitness.values = fit 

    for subpop_index, subpopulation in subpops_SP.items():      
        # Calculate slack feasibility and set fitness accordingly. This is also where the cost gets assigned as an attribute to the individual
        for idx, individual in enumerate(subpopulation):  # Use enumerate to get the index
            slack_validity = slack_feasibility(individual)
            if slack_validity:
                individual.fitness.values = individual.fitness.values
            else:
                individual.fitness.values = (0,)
                    
            # Print the required details in one line
            print(f"Feasibility: {slack_validity}, Fitness: {individual.fitness.values}, Cost: {individual.cost}, Values: {individual[:]}, Subpop: {subpop_index}, Ind: {idx + 1}")


    # Initialize containers to store the fitness statistics
    avg_fitness_per_gen = {i: [] for i in range(1, NUM_SUBPOPS + 1)}  # For average fitness (existing logic)
    highest_fitness_per_gen = {i: [] for i in range(1, NUM_SUBPOPS + 1)}  # For highest fitness
    highest_fitness_sum_per_gen = []  # For sum of highest fitness values across subpopulations

    best_fitness_sum = float('-inf')  # Start with a very low value
    best_individuals = []  # List to store the best individuals

    low = [b[0] for b in low_up_bound]
    up = [b[1] for b in low_up_bound]

    elite_selections = {}

    g = 0
    while g < GENERATIONS: 
        g += 1
        print(f"-- Generation {g} --")

        offspring = {}
        current_individuals = []  # Track individuals contributing to this generation's highest sum
        highest_fitness_sum = 0  # Initialize the sum of highest fitness for this generation

        for subpop_index, subpopulation in subpops_SP.items():
            # Compute and store fitness values, excluding (0,) fitness values
            fitness_values = [ind.fitness.values[0] for ind in subpopulation if ind.fitness.values[0] != 0]

            # Calculate the average fitness (existing logic)
            if fitness_values:
                avg_fitness = sum(fitness_values) / len(fitness_values)
            else:
                avg_fitness = 0
            avg_fitness_per_gen[subpop_index].append(avg_fitness)

            # Calculate the highest fitness
            if fitness_values:
                highest_fitness = max(fitness_values)
            else:
                highest_fitness = 0
            highest_fitness_per_gen[subpop_index].append(highest_fitness)

            # Add to the total highest fitness sum for this generation
            highest_fitness_sum += highest_fitness

            # Identify the individual(s) contributing to the highest fitness
            best_individual = min(
                (ind for ind in subpopulation if ind.fitness.values[0] == highest_fitness),
                key=lambda ind: getattr(ind, 'cost', float('inf'))  # Select based on cost
            )

            # Add this individual to current_individuals
            current_individuals.append({
                "subpop_index": subpop_index,
                "fitness": best_individual.fitness.values[0],
                "cost": getattr(best_individual, 'cost', 0),
                "values": list(best_individual)
            })

            # Select the next generation individuals
            # Preserve the top ~% as elites and select the rest through tournament selection
            elite_count = int(0.2 * len(subpopulation))
            elite_selections[subpop_index] = toolbox.elitism(subpopulation, elite_count)
            offspring[subpop_index] = (elite_selections[subpop_index] + toolbox.tournament(subpopulation, (len(subpopulation) - elite_count)))

            # Clone the selected individuals
            offspring[subpop_index] = list(map(toolbox.clone, offspring[subpop_index]))


            # Apply crossover
            for child1, child2 in zip(offspring[subpop_index][::2], offspring[subpop_index][1::2]): 
                if random.random() < CROSSR:  # Use updated crossover probability
                    toolbox.mate(child1, child2, indpb=0.5) 
                    del child1.fitness.values 
                    del child2.fitness.values
                    del child1.cost
                    del child2.cost 

            # Apply mutation
            for mutant in offspring[subpop_index]:
                if random.random() <= 1:
                    # Apply mutPolynomialBounded with shared bounds
                    mutant, = toolbox.mutbound(mutant, low=low, up=up, eta=ETAV, indpb=INDMUT)
                    mutant[:] = [max(0, val) for val in mutant]  # Ensure values are non-negative
                    
                    # Delete fitness to ensure re-evaluation
                    if hasattr(mutant.fitness, 'values'):
                        del mutant.fitness.values
                    
                    if hasattr(mutant.cost, 'values'):
                        del mutant.cost

        

        # Append the total highest fitness sum for this generation
        highest_fitness_sum_per_gen.append(highest_fitness_sum)

        # Check if the current highest fitness sum is the best we've seen
        if highest_fitness_sum > best_fitness_sum:
            best_fitness_sum = highest_fitness_sum
            best_individuals = current_individuals



        # Calculate slack feasibility and set fitness accordingly
        feasible_individuals = {subpop_index: [] for subpop_index in offspring.keys()}  # Dictionary format
        infeasible_individuals = {subpop_index: [] for subpop_index in offspring.keys()}

        for subpop_index, subpopulation in offspring.items():
            # Step 1: Calculate slack feasibility
            for idx, individual in enumerate(subpopulation):
                slack_validity = slack_feasibility(individual)

                if slack_validity:
                    feasible_individuals[subpop_index].append(individual)
                    print(f"Feasible - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {individual}, Fitness: {individual.fitness.values}")
                else:
                    # Replace infeasible individuals with elites
                    if elite_selections[subpop_index]:  # Ensure there are elites left for this subpopulation
                        replacement = elite_selections[subpop_index].pop(0)  # Take one elite from this subpopulation's selection
                        subpopulation[idx] = toolbox.clone(replacement)  # Replace with a clone of the elite
                        feasible_individuals[subpop_index].append(subpopulation[idx])  # Add to feasible
                        print(f"Replaced with Elite - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {subpopulation[idx]}, Fitness: {subpopulation[idx].fitness.values}")
                    else:
                        # If no elites are left, assign zero fitness
                        individual.fitness.values = (0,)
                        infeasible_individuals[subpop_index].append(individual)
                        print(f"Infeasible - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {individual}, Fitness: {individual.fitness.values}")


            
        # Step 2: Calculate centroids and fitness for feasible individuals
        if feasible_individuals:  # Ensure there are feasible individuals
            centroids_offspring = copy.deepcopy(centroidSP(feasible_individuals))
            fitness_SP_offspring = fitness(feasible_individuals, centroids_offspring)

            # Debug print: Check centroids and fitness before assignment
            print("Centroids for Offspring:", centroids_offspring)
            print("Fitness Calculated for Feasible Individuals:", fitness_SP_offspring)

            # Assign calculated fitness to feasible individuals
            for subpop_index, individuals in feasible_individuals.items():
                print(f"Assigning fitness to Subpopulation {subpop_index}:")  # Debug print
                if individuals:  # Ensure there are individuals to process
                    for idx, individual in enumerate(individuals):
                        print(f"Before Assignment - Ind: {idx + 1}, Fitness: {individual.fitness.values}")
                        individual.fitness.values = fitness_SP_offspring[subpop_index][idx]
                        print(f"After Assignment - Ind: {idx + 1}, Fitness: {individual.fitness.values}")
                else:
                    print(f"Warning: No feasible individuals in Subpopulation {subpop_index}")


        # Combine feasible and infeasible individuals to form the new offspring
        for subpop_index in offspring.keys():
            print(f"--- Debugging Subpopulation {subpop_index} ---")
            
            # Print counts of feasible and infeasible individuals
            feasible_count = len(feasible_individuals[subpop_index])
            infeasible_count = len(infeasible_individuals[subpop_index])
            print(f"Feasible Count: {feasible_count}, Infeasible Count: {infeasible_count}")

            # Validate total population size
            original_size = len(offspring[subpop_index])
            combined_size = feasible_count + infeasible_count
            print(f"Original Population Size: {original_size}, Combined Size After Merge: {combined_size}")
            assert combined_size == original_size, (
                f"Mismatch in population size for Subpopulation {subpop_index}: "
                f"Original: {original_size}, Combined: {combined_size}"
            )
            
            # Print a few individuals for validation
            print("Feasible Individuals:")
            for idx, ind in enumerate(feasible_individuals[subpop_index][:5], start=1):  # Print up to 5 feasible individuals
                print(f"  Ind {idx}: Fitness: {ind.fitness.values}, Values: {ind}")
            
            print("Infeasible Individuals:")
            for idx, ind in enumerate(infeasible_individuals[subpop_index][:5], start=1):  # Print up to 5 infeasible individuals
                print(f"  Ind {idx}: Fitness: {ind.fitness.values}, Values: {ind}")
            
            # Combine and update offspring
            offspring[subpop_index] = feasible_individuals[subpop_index] + infeasible_individuals[subpop_index]
            print(f"New Offspring Size: {len(offspring[subpop_index])}")
            print("-" * 40)

        # Step 3: Print generation summary for all subpopulations
        for subpop_index, subpopulation in offspring.items():
            for idx, individual in enumerate(subpopulation):
                cost = getattr(individual, 'cost', 'N/A')  # Safeguard for missing cost attribute
                print(f"Fitness: {individual.fitness.values}, Cost: {cost}, Values: {individual}, Subpop: {subpop_index}, Ind: {idx + 1}")
        print("-" * 40)

        # change parameters
        if g == value_change_1:
            ETAV = ETAV2
            INDMUT = INDMUT2
            print(f"Changing parameters (eta = {ETAV}).")

        if g == value_change_2:
            ETAV = ETAV3
            print(f"Changing parameters (eta = {ETAV}).")


        # change resolution
        if g == resolution_change_1:
            print("Switching to higher resolution. Resolution is 3.5D")
            model = model_3_5D

        # change resolution (full scale 6H)
        if g == resolution_change_2:
            print("Switching to higher resolution. Resolution is 6H")
            model = model_6H
        

        #assign offspring to subpops_SP so that created generation goes to the next loop
        subpops_SP = offspring

    end_time = time.time()
    elapsed_time = end_time - start_time


    # Display the best individuals after all generations
    print("\nBest Individuals Across All Generations:")
    print(f"Highest Fitness Sum: {best_fitness_sum}")
    for ind in best_individuals:
        print(f"  Subpopulation {ind['subpop_index']} - Fitness: {ind['fitness']:.2f}, "
            f"Cost: {ind['cost']:.2f}, Values: {ind['values']}")
    print(f"Elapsed time: {elapsed_time} seconds")



    
    # Line graph for highest fitness per subpopulation
    plt.figure(figsize=(10, 6))
    for subpop_index, fitness_values in highest_fitness_per_gen.items():
        plt.plot(range(1, len(fitness_values) + 1), fitness_values, label=f"Subpopulation {subpop_index}")

    # Add plot details
    plt.title(f"Highest Fitness per Generation (run {run})")
    plt.xlabel("Generation")
    plt.ylabel("Highest Fitness")
    plt.legend()
    plt.grid()

    file_path = os.path.join(results_dir, f"Highest_Fitness_Run_{run}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


    # Line graph for the sum of highest fitness values across subpopulations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(highest_fitness_sum_per_gen) + 1), highest_fitness_sum_per_gen, 
            label="Sum of Max Fitness", color='orange', linewidth=2)

    # Add plot details
    plt.title(f"Sum of Highest Fitness per Generation (Run {run})")
    plt.xlabel("Generation")
    plt.ylabel("Sum of Highest Fitness")
    plt.legend()
    plt.grid()


    file_path = os.path.join(results_dir, f"Sum_of_Highest_Fitness_Run_{run}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

 
    # Loop to generate and save the plot for each run
    plt.figure(figsize=(10, 5))
    for i in range(1, NUM_SUBPOPS + 1):
        plt.plot(avg_fitness_per_gen[i], label=f'Subpopulation {i} Avg Fitness')

    # Add plot details
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title(f'Average Fitness over Generations for Each Subpopulation (Run {run})')
    plt.legend()


    file_path = os.path.join(results_dir, f"Avg_Fitness_Per_Gen_Run_{run}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()




    # Extract technology labels dynamically from the `updates` list
    tech_labels = [f"{update['tech']} ({update['loc']})" for update in updates]

    initial_tech_labels = filtered_energy_cap_df_6H.index.tolist()
    initial_values = filtered_energy_cap_df_6H.values

    reformatted_tech_labels = [
        f"{label.split(' ')[1][1:-1]}::{label.split(' ')[0]}" for label in tech_labels
    ]

    # Now align the initial values
    aligned_initial_values = [
        initial_values[initial_tech_labels.index(label)] if label in initial_tech_labels else 0
        for label in reformatted_tech_labels
    ]

    
    # Adjust the number of bar groups
    num_bar_groups = len(best_individuals) + 1  # Number of subpopulations + initial capacities

    # Bar width and x locations
    x = np.arange(len(tech_labels))  # Number of technologies
    width = 0.8 / num_bar_groups  # Distribute bars within a single technology group

    # Plot each subpopulation's values
    plt.figure(figsize=(14, 7))

    # Plot initial capacities
    initial_bars = plt.bar(x - (num_bar_groups - 1) * width / 2, aligned_initial_values, width=width, label="Initial Capacities", color='gray')

    # Annotate initial capacities with rotated text
    for bar, value in zip(initial_bars, aligned_initial_values):
        if bar.get_height() > 0.2 * max(aligned_initial_values):  # Place inside for sufficiently large bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{value:.2f}",
                ha='center',
                va='center',
                rotation=45,
                fontsize=9,
                color='black'
            )
        else:  # Place above for small bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.2f}",
                ha='center',
                va='bottom',
                rotation=45,
                fontsize=9,
                color='black'
            )

    # Plot each subpopulation's values
    for idx, individual in enumerate(best_individuals, start=1):  # Iterate over the list
        values = individual['values']  # Extract the values for this individual
        bars = plt.bar(x - (num_bar_groups - 1) * width / 2 + idx * width, values, width=width, label=f"Subpopulation {individual['subpop_index']}")
        
        # Annotate the bars with values
        for bar, value in zip(bars, values):
            if bar.get_height() > 0.2 * max(values):  # Place inside for sufficiently large bars
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{value:.2f}",
                    ha='center',
                    va='center',
                    rotation=45,
                    fontsize=9,
                    color='black'
                )
            else:  # Place above for small bars
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{value:.2f}",
                    ha='center',
                    va='bottom',
                    rotation=45,
                    fontsize=9,
                    color='black'
                )

    # Add labels, title, and legend
    plt.xlabel("Technologies")
    plt.ylabel("Values (kW)")
    plt.title(f"Comparison of Best Individual Values and Initial Capacities (Run {run})")
    plt.xticks(x, tech_labels, rotation=45, ha="right")  # Align x-axis labels
    plt.legend(title="Subpopulations")
    plt.tight_layout()

    # Save the plot before showing it
    file_path = os.path.join(results_dir, f"Comparison_Run_{run}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()