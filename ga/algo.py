import subprocess

from deap import tools
from deap.algorithms import varAnd, varOr

from ga.utils import configs, get_map, logger, save_checkpoint, write_stds


def custom_eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None,
                   halloffame=None, verbose=__debug__) :
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
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
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, range(1, len(invalid_ind)+1), [gen]*len(invalid_ind))
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
    write_stds(process)
    # TODO: check loss and early stop
    # os.kill(process.pid, signal.SIGTERM); import signal
    exitcode = process.wait()
    # os.remove(config_path)

    if exitcode != 0:
        logger.error(f"Process {process.pid} failed with exit code {exitcode}")
        return 100000, 0
    else:
            # evaluate latency
            latency = 1000  # TODO: nvtx to get latency

    mAP = get_map(run_dir)
    return latency, mAP
