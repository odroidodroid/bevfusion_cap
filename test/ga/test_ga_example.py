import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


NGEN = 50
MU = 50
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.2

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20
random.seed(64)


class GA:
    def __init__(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", set, fitness=creator.Fitness)

        # Create the item dictionary: item name is an integer, and value is 
        # a (weight, value) 2-tuple.
        items = {}
        # Create random items and store them in the items' dictionary.
        for i in range(NBR_ITEMS):
            items[i] = (random.randint(1, 10), random.uniform(0, 100))
        self.items = items

        toolbox = base.Toolbox()
        toolbox.register("attr_item", random.randrange, NBR_ITEMS)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalKnapsack)
        toolbox.register("mate", self.cxSet)
        toolbox.register("mutate", self.mutSet)
        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox

    def evalKnapsack(self, individual):
        weight = 0.0
        value = 0.0
        for item in individual:
            weight += self.items[item][0]
            value += self.items[item][1]
        if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
            return 10000, 0             # Ensure overweighted bags are dominated
        return weight, value

    def cxSet(self, ind1, ind2):
        """Apply a crossover operation on input sets. The first child is the
        intersection of the two sets, the second child is the difference of the
        two sets.
        """
        temp = set(ind1)                # Used in order to keep type
        ind1 &= ind2                    # Intersection (inplace)
        ind2 ^= temp                    # Symmetric Difference (inplace)
        return ind1, ind2

    def mutSet(self, individual):
        """Mutation that pops or add an element."""
        if random.random() < 0.5:
            if len(individual) > 0:     # We cannot pop from an empty set
                individual.remove(random.choice(sorted(tuple(individual))))
        else:
            individual.add(random.randrange(NBR_ITEMS))
        return individual,
    
    def search(self):
        pop = self.toolbox.population(n=MU)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

        return pop, stats, hof


if __name__ == '__main__':
    ga = GA()
    hof = ga.search()
    print("done")
