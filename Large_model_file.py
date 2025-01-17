#!/usr/bin/env python3
# %%

import pandas as pd
import xarray as xr
import numpy as np
import copy
import math
import random
import warnings
import matplotlib.pyplot as plt
import os
from deap import base
from deap import creator
from deap import tools
import calliope
from calliope.exceptions import ModelWarning

calliope.set_log_verbosity(verbosity='critical', include_solver_output=False, capture_warnings=False)
# Suppress the specific ModelWarning from Calliope
#warnings.filterwarnings("ignore", category=ModelWarning)


# %%
#input variables
NUM_SUBPOPS = 2 # number of populations
SUBPOP_SIZE = 4 #size of populations. Must be even due to crossover
GENERATIONS = 100 #amount of generations
INDCROSS = 0.5 #chance for value of individual to be crossed over
INDMUT = 0.2 #chance for individual to mutate
ETAV = 100 #small value (1 or smaller) creates different values from parents, high value (20 or higher) creates resembling values


# changing values/resolution after n amount of generations
value_change_1 = 21  
value_change_2 = 60 

# change to other resolution after n amount of generations
resolution_change_1 = 90
resolution_change_2 = 95

# slack value
max_slack = 0.2

# Replace with a reasonable maximum if applicable this is already 1 = 1000GW
max_cap_value = 1 

# %%
#create the model with a resolution of 14 days
model_14D = calliope.Model('../model/euro_calliope/eurospores/model.yaml', scenario='config_overrides,res_14D,link_cap_dynamic,freeze-hydro-capacities,max_energy_caps')
model_14D.run()

df_total_cost_14D = model_14D.results.cost.to_series().dropna()
total_cost_optimal_14D = df_total_cost_14D.loc[~df_total_cost_14D.index.map(str).str.contains('co2_emissions')].sum()

energy_cap_df_14D = model_14D.results.energy_cap.to_pandas()
filtered_energy_cap_df_14D = energy_cap_df_14D[~energy_cap_df_14D.index.str.contains("demand|transmission")]

#create the model with a resolution of 14 days
model_5D = calliope.Model('../model/euro_calliope/eurospores/model.yaml', scenario='config_overrides,res_5D,link_cap_dynamic,freeze-hydro-capacities,max_energy_caps')
model_5D.run()

df_total_cost_5D = model_5D.results.cost.to_series().dropna()
total_cost_optimal_5D = df_total_cost_5D.loc[~df_total_cost_5D.index.map(str).str.contains('co2_emissions')].sum()

energy_cap_df_5D = model_5D.results.energy_cap.to_pandas()
filtered_energy_cap_df_5D = energy_cap_df_5D[~energy_cap_df_5D.index.str.contains("demand|transmission")]

#create the model with a resolution of 14 days
model_FULL = calliope.Model('../model/euro_calliope/eurospores/model.yaml', scenario='config_overrides,res_6H,link_cap_dynamic,freeze-hydro-capacities,max_energy_caps')
model_FULL.run()

df_total_cost_FULL = model_FULL.results.cost.to_series().dropna()
total_cost_optimal_FULL = df_total_cost_FULL.loc[~df_total_cost_FULL.index.map(str).str.contains('co2_emissions')].sum()

energy_cap_df_FULL = model_FULL.results.energy_cap.to_pandas()
filtered_energy_cap_df_FULL = energy_cap_df_FULL[~energy_cap_df_FULL.index.str.contains("demand|transmission")]



# %%
# Put the capacities in a list
initial_capacities_FULL = filtered_energy_cap_df_FULL.values 

# if a value is lower than 13-5, set the value to 0. This is due to everything below 1e-5 being so low it it not interesting to include
initial_capacities_FULL[initial_capacities_FULL < 1e-5] = 0

# adjust when using more models
optimal_value = total_cost_optimal_FULL # adjust when using more models

model = model_14D


# %%
#Order of technologies is stored. This is to know later on which position in the capacity list corresponds to which technology
updates = [
    {'tech': tech, 'loc': loc}
    for loc_tech in filtered_energy_cap_df_FULL.index
    for loc, tech in [loc_tech.split("::")]  # Split index by '::' to separate loc and tech
]

# %%

# Look in the model backend at what the values are that it has used to initialize the model
input_params = model.backend.access_model_inputs()

# Access energy_cap_max and energy_cap_min for each technology
energy_cap_max = input_params['energy_cap_max']
energy_cap_min = input_params['energy_cap_min']

# Convert to DataFrame for filtering
energy_cap_max_df = energy_cap_max.to_dataframe()
energy_cap_min_df = energy_cap_min.to_dataframe()

# Filter out rows with 'demand' or 'free' in the index
energy_cap_max_filtered = energy_cap_max_df[~energy_cap_max_df.index.get_level_values('loc_techs').str.contains("demand|transmission")]
energy_cap_min_filtered = energy_cap_min_df[~energy_cap_min_df.index.get_level_values('loc_techs').str.contains("demand|transmission")]


# Create a dictionary of loc_tech to [min, max] bounds meaning that you will have the loc_tech and their corresponding min max capacity
low_up_mapping = {
    loc_tech: [
        energy_cap_min_filtered.loc[loc_tech, 'energy_cap_min'],
        energy_cap_max_filtered.loc[loc_tech, 'energy_cap_max']
    ]
    for loc_tech in energy_cap_max_filtered.index
}

# Structure the updates (previously done, which are the tech:, loc: pairs) so that it can eventually be used to put in the backend
updates_order = [f"{update['loc']}::{update['tech']}" for update in updates]

# Put it in such a structure that it can be used for the mutation function. This wants it in a list order
low_up_bound = [
    low_up_mapping[loc_tech] for loc_tech in updates_order
]

# Check for 'inf' in upper bounds and adjust if needed
for i, (low, up) in enumerate(low_up_bound):
    if up == float('inf'):
        low_up_bound[i][1] = max_cap_value


# %%
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

        total_cost = initial_cost
    
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
    feasible = slack_distance <= max_slack and cost >= optimal_value
    
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

# Euclidean distance
def fitness(subpop, centroids):
    distances = []
    minimal_distances = []
    fitness_SP = {}

    # Step 1: Calculate Euclidean Distances for each individual
    for q, (subpop_index, subpopulation) in enumerate(subpop.items()):
        subpopulation_distances = []
        
        for individual in subpopulation:
            individual_distances = []
            
            for p, centroid in enumerate(centroids):
                if p != q:  # Skip the centroid of the same subpopulation
                    # Calculate Euclidean distance
                    distance = math.sqrt(sum((individual[i] - centroid[i])**2 for i in range(len(individual))))
                    individual_distances.append(distance)
            
            subpopulation_distances.append(individual_distances)
        
        distances.append(subpopulation_distances)

    # Step 2: Calculate Minimal Distances
    for subpopulation_distances in distances:
        subpopulation_minimal = [min(individual_distances) for individual_distances in subpopulation_distances]
        minimal_distances.append(subpopulation_minimal)

    # Step 3: Calculate Fitness SP for each individual
    for sp_index, subpopulation in enumerate(minimal_distances, start=1):
        fitness_values = [(min_distance,) for min_distance in subpopulation]
        fitness_SP[sp_index] = fitness_values

    return fitness_SP

#  Absolute distance
def fitness_abs(subpop, centroids):
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
    adjusted_individual = []
    
    for cap, (low, up) in zip(initial_capacities_FULL, low_up_bound):
        if cap == 0:
            # Small chance for installation between % of upper bound
            if random.random() < 0.2:  # 10% chance
                new_value = random.uniform(0.00001 * up, 0.0001 * up)
            else:
                # No fallback value, skip adjustment
                new_value = 0.0  # Keeps as zero or explicitly sets to 0
        else:
            # Adjust by a random value between -0.1 and 0.1 of the current value
            adjustment = random.uniform(0, 0.01)
            new_value = cap * (1 + adjustment)
        
        # Ensure the new value is within the lower and upper bounds
        new_value = max(low, min(up, new_value))
        adjusted_individual.append(new_value)
    
    return adjusted_individual


# %%
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

# Generate subpopulations with multiple individuals
subpops_unaltered = [toolbox.subpopulationSP(n=SUBPOP_SIZE) for _ in range(NUM_SUBPOPS)]

subpops_SP = {}

for p in range(NUM_SUBPOPS):
    subpops_SP[p+1] = subpops_unaltered[p]

# %%
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
        print(f"Feasibility: {slack_validity}, Fitness: {individual.fitness.values}, Cost: {individual.cost}, Subpop: {subpop_index}, Ind: {idx + 1}")

# %%
##################### initialize all the containers and variables ######################################

# Initialize containers to store the fitness statistics
highest_fitness_per_gen = {i: [] for i in range(1, NUM_SUBPOPS + 1)}  # For highest fitness
highest_fitness_sum_per_gen = []  # For sum of highest fitness values across subpopulations

best_fitness_sum = float('-inf')  # Start with a very low value
best_individuals = []  # List to store the best individuals

# Initialize a DataFrame for tracking Best Individuals across generations, including loc_tech columns
best_individuals_columns = ["generation", "subpopulation", "fitness", "cost", "values"] + updates_order
best_individuals_df = pd.DataFrame(columns=best_individuals_columns)

# State the low and upper bounds for the mutation
low = [b[0] for b in low_up_bound]
up = [b[1] for b in low_up_bound]

# Initialize dictionaries to store the elite selections and ranked list of individuals
elite_selections = {}
ranked_list_ind = {}

# Initialize the generation export buffer
generation_export_buffer = []

# Initialize a flag to track the first resolution change
first_resolution_change = False
# Initialize buffers to store individuals that are feasible after resolution change
feasible_after_resolution_buffer = []

#make sure file does not exists and if so, delete it
filename2 = "feasible_after_resolution.xlsx"
sheet_name = "Feasible_Individuals"
if os.path.exists(filename2):
    os.remove(filename2)

# %%
g = 0
while g < GENERATIONS: 
    g += 1
    print(f"-- Generation {g} --")

    offspring = {}
    current_individuals = []  # Track individuals contributing to this generation's highest sum
    highest_fitness_sum = 0  # Initialize the sum of highest fitness for this generation
    export_data = []

    for subpop_index, subpopulation in subpops_SP.items():
        # Compute and store fitness values, excluding (0,) fitness values
        fitness_values = [ind.fitness.values[0] for ind in subpopulation if ind.fitness.values[0] != 0]

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
            "generation": g,  # Add the generation
            "values": list(best_individual)
        })




############################################## operator code ##############################################

        #Rank the individuals for if they are needed when some mutated individuals are deemed infeasible
        ranked_list_ind[subpop_index] = toolbox.elitism(subpopulation, len(subpopulation))
        # Select the next generation individuals
        # Preserve the top ~% as elites and select the rest through tournament selection
        elite_count = int(0.2 * len(subpopulation))
        elite_selections[subpop_index] = toolbox.elitism(subpopulation, elite_count)
        offspring[subpop_index] = (elite_selections[subpop_index] + toolbox.tournament(subpopulation, (len(subpopulation) - elite_count)))
        # Clone the selected individuals
        offspring[subpop_index] = list(map(toolbox.clone, offspring[subpop_index]))

        # Apply crossover
        for child1, child2 in zip(offspring[subpop_index][::2], offspring[subpop_index][1::2]): 
            if random.random() < 0.5:  # Use updated crossover probability
                toolbox.mate(child1, child2, indpb=INDCROSS) 
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

############################################## storing best individual ##############################################

    # Append the total highest fitness sum for this generation
    highest_fitness_sum_per_gen.append(highest_fitness_sum)


    # Track Best Individuals and Update DataFrame
    if highest_fitness_sum > best_fitness_sum:
        best_fitness_sum = highest_fitness_sum
        best_individuals = current_individuals.copy()  # Update the best individuals

        # Add new best individuals directly to the DataFrame so they can eventually be exported to the excel file
        for ind in best_individuals:
            row = {
                "generation": ind['generation'],
                "subpopulation": ind['subpop_index'],
                "fitness": ind['fitness'],
                "cost": ind['cost'],
                "values": ', '.join(map(str, ind['values']))
            }
            
            # Add loc_tech values from the individual
            for loc_tech, value in zip(updates_order, ind['values']):
                row[loc_tech] = value
            
            # Append the row to the DataFrame
            best_individuals_df = pd.concat([best_individuals_df, pd.DataFrame([row])], ignore_index=True)

############################################## calculating centroids and feasibility ##############################################

    # Calculate slack feasibility and set fitness accordingly
    feasible_individuals = {subpop_index: [] for subpop_index in offspring.keys()}  
    infeasible_individuals = {subpop_index: [] for subpop_index in offspring.keys()}

    for subpop_index, subpopulation in offspring.items():
        
        # Step 1: Calculate slack feasibility
        for idx, individual in enumerate(subpopulation):
            slack_validity = slack_feasibility(individual)

            if slack_validity:
                feasible_individuals[subpop_index].append(individual)
                # save the individuals that are feasible after the first resolution change has been initialized
                if first_resolution_change:  
                        # Store the feasible individual in the buffer
                        row = {
                            "generation": g,
                            "subpopulation": subpop_index,
                            "individual": f"Subpop {subpop_index}, Ind {idx + 1}",
                            "cost": getattr(individual, 'cost', 'N/A')
                        }
                        
                        # Add loc_tech values
                        for loc_tech, value in zip(updates_order, individual):
                            row[loc_tech] = value
                        
                        feasible_after_resolution_buffer.append(row)
                    
            else:
                # Replace infeasible individuals with elites
                if elite_selections[subpop_index]:  # Ensure there are elites left for this subpopulation
                    replacement = elite_selections[subpop_index].pop(0)  # Take one elite from this subpopulation's selection
                    subpopulation[idx] = toolbox.clone(replacement)  # Replace with a clone of the elite
                    feasible_individuals[subpop_index].append(subpopulation[idx])  # Add to feasible
                    print(f"Replaced with Elite - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {subpopulation[idx]}, Fitness: {subpopulation[idx].fitness.values}")

                # # Replace with a random feasible individual from ranked_list_ind    
                elif ranked_list_ind[subpop_index]:
                    replacement = random.choice(ranked_list_ind[subpop_index])
                    ranked_list_ind[subpop_index].remove(replacement)  # Remove the selected individual
                    subpopulation[idx] = toolbox.clone(replacement)
                    feasible_individuals[subpop_index].append(subpopulation[idx]) # add to infeasible
                    print(f"Replaced with one previous feasible individual - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {subpopulation[idx]}, Fitness: {subpopulation[idx].fitness.values}")

                # Assign zero fitness if no replacements are available
                else:
                    # If no elites or previous fit values are left, assign zero fitness and continue
                    individual.fitness.values = (0,)
                    infeasible_individuals[subpop_index].append(individual)
                    print(f"Infeasible - Subpop: {subpop_index}, Ind: {idx + 1}, Values: {individual}, Fitness: {individual.fitness.values}")

 
    # Step 2: Calculate centroids and fitness for feasible individuals
    if feasible_individuals:  
        centroids_offspring = copy.deepcopy(centroidSP(feasible_individuals))
        fitness_SP_offspring = fitness(feasible_individuals, centroids_offspring)

        # Assign calculated fitness to feasible individuals
        for subpop_index, individuals in feasible_individuals.items():
            if individuals:  # Ensure there are individuals to process
                for idx, individual in enumerate(individuals):
                    individual.fitness.values = fitness_SP_offspring[subpop_index][idx]
            else:
                print(f"Warning: No feasible individuals in Subpopulation {subpop_index}")

    # Combine feasible and infeasible individuals to form the new offspring and make sure the original and combined sizes match
    for subpop_index in offspring.keys():
        
        feasible_count = len(feasible_individuals[subpop_index])
        infeasible_count = len(infeasible_individuals[subpop_index])

        # Validate total population size
        original_size = len(offspring[subpop_index])
        combined_size = feasible_count + infeasible_count
        
        assert combined_size == original_size, (
            f"Mismatch in population size for Subpopulation {subpop_index}: "
            f"Original: {original_size}, Combined: {combined_size}"
        )
        
        # Combine and update offspring
        offspring[subpop_index] = feasible_individuals[subpop_index] + infeasible_individuals[subpop_index]


############################################## value and resolution changes ##############################################

    # change parameters and or resolution after n generations
    if g == value_change_1:
        print("Changing parameters (eta = 1000).")
        ETAV = 1000

    if g == value_change_2:
        print("Changing parameters (eta = 10000).")
        ETAV = 10000

    if g == resolution_change_1:
        print("Changing resolution.")
        first_resolution_change = True
        model = model_5D
    
    if g == resolution_change_2:
        print("Changing resolution to 6H.")
        model = model_FULL

############################################## Make the offspring the new subpops_SP ######################################

    # Update the subpopulations with the new offspring
    subpops_SP = offspring



############################################## code to export to excel ####################################################

    # Export data for specific generations to excel
    generation_export_buffer.clear()

    for subpop_index, subpopulation in subpops_SP.items():
        for idx, individual in enumerate(subpopulation):
            if individual.fitness.values[0] == 0:
                continue  # Skip individuals with zero fitness
            
            row = {
                "generation": g,
                "subpopulation": subpop_index,
                "individual": f"Subpop {subpop_index}, Ind {idx + 1}",
                "fitness": individual.fitness.values[0],
                "cost": getattr(individual, 'cost', 'N/A')
            }
            
            for loc_tech, value in zip(updates_order, individual):
                row[loc_tech] = value  # Map loc::tech values
            
            generation_export_buffer.append(row)

    # Export data for specific generations
    if g == 1 or g % 10 == 0 or g == GENERATIONS:
        if generation_export_buffer:
            # Convert buffer to DataFrame
            df_gen = pd.DataFrame(generation_export_buffer)
            
            # Sort the data for clarity
            df_gen = df_gen.sort_values(
                by=["subpopulation", "fitness"], 
                ascending=[True, False]
            ).reset_index(drop=True)
            
            # Write to an Excel file with a separate sheet for each generation
            filename = "individual_generation_interval.xlsx"
            sheet_name = f"Generation_{g}"
            
            # Remove the file if it exists
            if g == 1 and os.path.exists(filename):
                os.remove(filename) 

            # Open Excel file in append mode or create if it doesn't exist
            try:
                with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
                    df_gen.to_excel(writer, sheet_name=sheet_name, index=False)
            except FileNotFoundError:
                with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                    df_gen.to_excel(writer, sheet_name=sheet_name, index=False)



############################ code to export feasible individuals after resolution change ##############################################
      
    # Check if the buffer has feasible individuals
    if feasible_after_resolution_buffer:
        # Convert buffer to DataFrame
        df_feasible = pd.DataFrame(feasible_after_resolution_buffer)

        # Define the new file name and sheet name
        filename2 = "feasible_after_resolution.xlsx"
        sheet_name = "Feasible_Individuals"

        # Check if the file exists
        if not os.path.exists(filename2):
            # If the file doesn't exist, create a new one
            with pd.ExcelWriter(filename2, mode='w', engine='openpyxl') as writer:
                df_feasible.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"File '{filename2}' created with data.")
        else:
            # Append data to the existing file
            with pd.ExcelWriter(filename2, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df_feasible.to_excel(writer, sheet_name=sheet_name, index=False, startrow=writer.sheets[sheet_name].max_row)
            print(f"Data appended to '{filename2}'.")

        # Clear the buffer after exporting
        feasible_after_resolution_buffer.clear()

# %%
# Export Best Individuals to Excel after the loop finishes
filename = "individual_generation_interval.xlsx"
try:
    with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
        best_individuals_df.to_excel(writer, sheet_name="Best_Individuals", index=False)
except FileNotFoundError:
    with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
        best_individuals_df.to_excel(writer, sheet_name="Best_Individuals", index=False)

# %%
print("\nBest Individuals Across All Generations:")
print(f"Highest Fitness Sum: {best_fitness_sum}")
for ind in best_individuals:
    print(f"  Generation {ind['generation']} - Subpopulation {ind['subpop_index']} - "
        f"Fitness: {ind['fitness']:.2f}, Cost: {ind['cost']:.2f}, Values: {ind['values']}")


