
import numpy as np
import random

def genetic_algorithm_searcher(search_space: SearchSpace, population_size,
                               min_objective=True, mutation_p, cross_over_p):

    def population_initializer(p_size):
        return search_space.generate_population(p_size)

    def mutation_function(x):
        
        return mutation(x) # TODO: implement mutation function

    def cross_over_function(x1, x2):
        gene_length = search_space.gene_length
        point1 = random.randint(0, gene_length-1)
        point2 = random.randint(point1, gene_length)
        child1 = np.concatenate(np.concatenate(x1[:point1], x2[point1:point2]]), x1[point2:])
        child2 = np.concatenate(np.concatenate(x2[:point2], x1[point1:point2]]), x2[point2:])
            
        return (child1, child2)
    
    def selection_function(x):
        probabilities=fitness_scores # TODO: implement fitness score / function
        choices=np.random.choice(numbers,p=probabilities/probabilities.sum())
        return choice
        
    return GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function,
                             min_objective=min_objective,
                             population_size=population_size)

class SearchSpace(object):
    def __init__(self, configs):
        self.cfg = configs
        # ResNet: Depth, # of channel, kernel size
        # LSS: Grid size, image resolution, depth distribution lenghth, depth distribution interval, BEV feature channel size
    
    def generate_vector(self):
        x = []
        for i in configs:
            x.append(random.choice(i))

        return np.array(x)
    
    def gene_length(self):
        return len(self.generate_vector())

    def generate_population(self, size):
        return [self.generate_vector() for _ in range(size)]
    

class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size):
        # Functions
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        # parameters
        self.population_size = population_size

    def create_new_generation(self, population):
        new_generation = []
        couples = self.selection_function(population) # selection
        for i in couples:
            children = self.cross_over_function(i[0], i[1]) # cross-over
            for j in children:
                j = self.mutation_function(j) # mutation
        for c in children: 
            new_generation.append(c)
        
        return new_generation

    def get_current_generation(self):
        return self.generation


