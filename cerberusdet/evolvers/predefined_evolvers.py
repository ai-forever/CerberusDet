# https://docs.ray.io/en/latest/tune/api/suggestion.html
EVOLVER_TYPES = [
    # Uses https://ax.dev/ to optimize hyps
    # [ray.tune.search.ax.ax_search.AxSearch]
    "ax",  # pip install ax-platform sqlalchemy
    # wrapper around Optuna
    # [ray.tune.search.optuna.OptunaSearch]
    "optuna",  # pip install optuna
    # uses Bayesian Optimization to improve the hyperparameter search
    # [ray.tune.search.bohb.TuneBOHB]
    "bohb",  # pip install hpbandster ConfigSpace
    # search algorithm based on randomized local search [allows  to specify a low-cost initial point as input]
    # [ray.tune.search.flaml.CFO]
    "cfo",  # pip install flaml
    # Uses Dragonfly to optimize hyps
    # [ray.tune.search.dragonfly.DragonflySearch]
    "dragonfly",  # pip install dragonfly-opt
    # Heteroscedastic Evolutionary Bayesian Optimization
    # [ray.tune.search.hebo.HEBOSearch]
    # "hebo",                                   # pip install HEBO>=0.2.0 (?)
    # Uses Nevergrad to optimize hyps
    # [ray.tune.search.nevergrad.NevergradSearch]
    "nevergrad",  # pip install nevergrad
    # Scikit Optimize (skopt)
    # [ray.tune.search.skopt.SkOptSearch]
    "skopt",  # pip install scikit-optimize
    # A wrapper around ZOOpt
    # [ray.tune.search.zoopt.ZOOptSearch]
    "zoopt",  # pip install zoopt
    # [default] do hyps search via random and grid search
    # [ray.tune.search.basic_variant.BasicVariantGenerator]
    "random",
]
