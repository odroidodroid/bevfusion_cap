import subprocess

from deap import tools
from deap.algorithms import varAnd, varOr

from ga.utils import configs, get_map, get_latency, logger, save_checkpoint, write_stds


def custom_eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None,
                   halloffame=None, verbose=__debug__) :
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    invalid_ind_len = len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, range(invalid_ind_len), [0]*invalid_ind_len)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
            
    # Begin the generational process
    for gen in range(1, ngen + 1):

        logger.info(f"================== Generation {gen} ==================")

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind_len = len(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, range(invalid_ind_len), [gen]*invalid_ind_len)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        save_checkpoint(population, halloffame, logbook, gen)

        logger.info(f"============ Best individual is {halloffame[0]} ============")
        logger.info(f"============ with fitness: {halloffame[0].fitness.values} ============")

    return population, logbook

def run_model(config_path, run_dir) :
    
    run = [
        "torchpack",
        "dist-run",
        f"-np={configs.NUM_GPUS}",
        "python",
        "tools/train.py",
        config_path,
        "--load_from=pretrained/lidar-only-det.pth",
        f"--run-dir={run_dir}",
    ]
    conda_env = configs.CONDA_ENV
    if conda_env is not None :
        conda_activate = f"conda activate {conda_env}"
        cmd = f"{conda_activate}; {' '.join(run)}"
    else :
        cmd = f"{' '.join(run)}"
    logger.info(f"{cmd=}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    logger.info(f"Started process {process.pid}")
    write_stds(process)
    # TODO: check loss and early stop
    # os.kill(process.pid, signal.SIGTERM); import signal
    exitcode = process.wait()
    # os.remove(config_path)

    if exitcode != 0:
        logger.error(f"Process {process.pid} failed with exit code {exitcode}")
        return 100000.0, 0.0
    else:
        # evaluate latency
        batch_size = 1
        exitcode = subprocess.run(
            f"nsys profile -y 6 -t cuda,nvtx -o {run_dir}/report1 --stats=true \
            python tools/test.py {config_path} \
            --data.samples_per_gpu={batch_size} --data.workers_per_gpu={batch_size} \
            {run_dir}/latest.pth --eval bbox --disable_dist",
            shell=True
        ).returncode

        # get latency
        latency = get_latency(run_dir)
        #get mAP
        mAP = get_map(run_dir)
        
    return latency, mAP
