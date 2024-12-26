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
                      chromosome_to_config_dict, crossover_twopoint,
                      generate_chromosome, mutate_onepoint)
from ga.utils import (configs, deep_update, get_latency, get_map, load_cache,
                      load_checkpoint, logger, save_cache, save_checkpoint,
                      write_stds)

random.seed(64)

class GA:
    def __init__(self, name: str = "untitled"):
        self.name = name

        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", dict, fitness=creator.Fitness)

        toolbox = base.Toolbox()
        # toolbox.register("attr_item", generate_chromosome)
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_chromosome)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalKnapsack)
        toolbox.register("mate", crossover_twopoint)
        toolbox.register("mutate", mutate_onepoint)
        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox
        self.cache_path = os.path.join("./ga/results", f"{name}_cache.json")
        self.cache = load_cache(self.cache_path)

    def evalKnapsack(self, individual, ind_idx, gen_idx):
        logger.info(f"================== Individual {ind_idx} ==================")
        logger.info(f"{individual=}")

        if str(individual) in self.cache:
            logger.info(f"Individual {ind_idx} already evaluated")
            latency = self.cache[str(individual)]['latency']
            mAP = self.cache[str(individual)]['mAP']
            logger.info(f"gen {gen_idx} ind {ind_idx} mAP: {mAP:.2f} latency: {latency:.2f}")
            logger.info(f"===================================================")
            return latency, mAP

        # make dir with gen and individual id(idx of population)
        run_dir = os.path.join(configs.PROJECT_DIR, f"gen_{gen_idx}", f"ind_{ind_idx}")
        os.makedirs(run_dir, exist_ok=True)

        config_dict = chromosome_default_resnet()
        deep_update(config_dict, chromosome_to_config_dict(individual))
        # deep_update(config_dict['data'], chromosome_minidataset())
        config_path = os.path.join(run_dir, f"{self.name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        latency, mAP = run_model(config_path, run_dir)

        logger.info(f"gen_{gen_idx}_ind_{ind_idx} mAP: {mAP:.2f} latency: {latency:.2f}")
        self.cache[str(individual)] = {"latency": latency, "mAP": mAP}
        save_cache(self.cache_path, self.cache)
        logger.info(f"===================================================")
        return latency, mAP

    def search(self, checkpoint=None):
        # generate initial population
        if checkpoint:
            pop, hof, logbook, gen = load_checkpoint(checkpoint)
        else:
            gen = 0
            pop = self.toolbox.population(n=configs.MU)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean, axis=0)
            stats.register("std", numpy.std, axis=0)
            stats.register("min", numpy.min, axis=0)
            stats.register("max", numpy.max, axis=0)
            logbook = None

        pop, logbook = custom_eaMuPlusLambda(
            pop, self.toolbox, configs.MU, configs.LAMBDA, configs.CXPB, configs.MUTPB, configs.NGEN,
            startgen=gen, stats=stats, halloffame=hof, logbook=logbook, verbose=True)

        return hof[0]


if __name__ == '__main__':
    ga = GA("bevfusion")
    try:
        best = ga.search()
    except KeyboardInterrupt as e:
        logger.info("============ Search has been terminated by user ============")
    else:
        logger.info("============ Search has been completed ============")
        logger.info("============ Best individual ============")
        logger.info(best)
    finally:
        logger.info(f"============= Time: {datetime.now()} =============")
    print("done")
