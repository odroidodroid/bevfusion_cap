import os
import random
import subprocess
from datetime import datetime

import numpy
import yaml
from deap import algorithms, base, creator, tools

from ga.genes import (chromosome_default_resnet, chromosome_minidataset,
                      chromosome_to_config_dict, generate_chromosome)
from ga.utils import (configs, deep_update, get_map, load_checkpoint, logger,
                      save_checkpoint, write_stds)
from ga.algo import run_model, custom_eaSimple
random.seed(64)


class GA:
    def __init__(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", dict, fitness=creator.Fitness)

        toolbox = base.Toolbox()
        # toolbox.register("attr_item", generate_chromosome)
        toolbox.register("individual", tools.initIterate,
                         creator.Individual, generate_chromosome)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("evaluate", self.evalKnapsack)
        toolbox.register("mate", self.cxSet)
        toolbox.register("mutate", self.mutSet)
        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox

    def evalKnapsack(self, individual):
        logger.info(f"================== Individual {self.current_ind} ==================")
        logger.info(f"{individual=}")

        # make dir with gen and individual id(idx of population)
        name = f"gen_{self.current_gen}_ind_{self.current_ind}"
        run_dir = os.path.join(configs.PROJECT_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        config_dict = chromosome_default_resnet()
        deep_update(config_dict, chromosome_to_config_dict(individual))
        deep_update(config_dict['data'], chromosome_minidataset())
        config_path = os.path.join(run_dir, f"{name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        # train individual
        latency, mAP = run_model(config_path, run_dir)

        # save results
        ind_results = {
            'chromosome' : individual,
            'latency' : latency,
            'accuracy' : mAP
        }        
        config_dict.update(ind_results)
        with open(config_path, 'w') as f :
            yaml.dump(config_dict, f)

        self.current_ind += 1

        logger.info(f"gen_{self.current_gen}_ind_{self.current_ind} mAP: {mAP} latency: {latency}")
        logger.info(f"======================================================")
        return latency, mAP

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
        """Mutation that change one gene in the set."""

        return individual,

    def search(self):
        # generate initial population
        pop = self.toolbox.population(n=configs.MU)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        # which one is better?
        # algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
        for gen in range(configs.NGEN):
            logger.info(f"================== Generation {gen} ==================")
            self.current_gen = gen
            self.current_ind = 0

            pop, logbook = custom_eaSimple(
                pop, self.toolbox, cxpb=configs.CXPB, mutpb=configs.MUTPB, ngen=1, stats=stats, halloffame=hof, verbose=True
            )

            logger.info(f"============ Best individual is {hof[0]} ============")
            logger.info(f"============ with fitness: {hof[0].fitness.values} ============")

        return hof[0]


if __name__ == '__main__':
    ga = GA()
    try:
        best = ga.search()
    except KeyboardInterrupt as e:
        logger.info("============ Search has been terminated by user ============")
    finally:
        logger.info("============ Search has been terminated ============")
        logger.info("============ Best individual is ============")
        logger.info(best)
        logger.info(f"============ with fitness: {best.fitness.values} ============")
        logger.info("============ Search has been terminated ============")
        logger.info(f"============ Time: {datetime.now()} ============")
    print("done")
