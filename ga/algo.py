from deap import tools
from deap.algorithms import varAnd
import subprocess
from ga.utils import write_stds, logger, get_map

def custom_eaSimple(population, toolbox, cxpb, mutpb, ngen, k=None, stats=None,
                   halloffame=None, verbose=__debug__) :
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, population) ## 수정
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, k)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = population + offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        
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

    return get_map(run_dir), latency
