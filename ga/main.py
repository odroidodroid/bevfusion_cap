import json
import os
import random
import subprocess
from datetime import datetime

import numpy
import yaml
from deap import algorithms, base, creator, tools

from ga.algo import custom_eaMuPlusLambda, run_model
from ga.genes import (chromosome_default_resnet, chromosome_minidataset,
                      chromosome_to_config_dict, generate_chromosome)
from ga.utils import (configs, deep_update, get_map, load_checkpoint, logger,
                      save_checkpoint, save_results, write_stds)

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

    def evalKnapsack(self, individual, ind_idx, gen_idx):
        logger.info(f"================== Individual {ind_idx} ==================")
        logger.info(f"{individual=}")

        # make dir with gen and individual id(idx of population)
        name = f"gen_{gen_idx}_ind_{ind_idx}"
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
            'gen_idx' : gen_idx,
            'ind_idx' : ind_idx,
            'latency' : latency,
            'accuracy' : mAP,
            'chomosome' : individual
        }
        
        save_results(configs.RESULT_FILE_PATH, ind_results)

        logger.info(f"gen_{gen_idx}_ind_{ind_idx} mAP: {mAP} latency: {latency}")
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
        pop, logbook = custom_eaMuPlusLambda(pop, 
                                             self.toolbox, 
                                             mu=configs.MU,
                                             lambda_=configs.LAMBDA,
                                             cxpb=configs.CXPB, 
                                             mutpb=configs.MUTPB, 
                                             ngen=configs.NGEN, 
                                             stats=stats, 
                                             halloffame=hof, 
                                             verbose=True)

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
