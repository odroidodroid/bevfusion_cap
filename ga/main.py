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
from ga.utils import (configs, deep_update, get_latency, get_map,
                      load_checkpoint, logger, save_checkpoint, write_stds)

random.seed(64)

class GA:
    def __init__(self):
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

        conda_env = "bevfusion"
        conda_activate = f"conda activate {conda_env}"
        run = [
            "torchpack",
            "dist-run",
            "-np=2",
            "python",
            "tools/train.py",
            config_path,
            "--load_from=pretrained/lidar-only-det.pth",
            f"--run-dir={run_dir}",
        ]
        cmd = f"{conda_activate}; {' '.join(run)}"
        logger.info(f"{cmd=}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        logger.info(f"Started process {process.pid}")
        write_stds(process, enableStderr=True)
        # TODO: check loss and early stop
        # os.kill(process.pid, signal.SIGTERM); import signal
        exitcode = process.wait()

        self.current_ind += 1
        if exitcode != 0:
            logger.error(f"Process {process.pid} failed with exit code {exitcode}")
            latency, mAP = 10000, 0.0
        else:
            # evaluate latency
            batch_size = 1
            exitcode = subprocess.run(
                f"nsys profile -y 6 -t cuda,nvtx -o {run_dir}/report1 --stats=true \
                python tools/test.py {config_path} --data.samples_per_gpu={batch_size} --data.workers_per_gpu={batch_size} \
                {run_dir}/latest.pth --eval bbox --disable_dist",
                shell=True
            ).returncode

            # get latency
            latency = get_latency(run_dir)

            # get mAP
            mAP = get_map(run_dir)

        logger.info(f"gen_{self.current_gen}_ind_{self.current_ind} mAP: {mAP:.2f} latency: {latency:.2f}")
        logger.info(f"===================================================")
        return latency, mAP

    def search(self):
        # generate initial population
        pop = self.toolbox.population(n=configs.MU)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

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
    else:
        logger.info("============ Search has been completed ============")
        logger.info("============ Best individual ============")
        logger.info(best)
    finally:
        logger.info(f"============ Time: {datetime.now()} ============")
    print("done")
