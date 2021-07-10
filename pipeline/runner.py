from functools import partial
from datetime import datetime
from functools import wraps
import pandas as pd
import numpy as np
import inspect
import pickle
import re
import os

cache_folder = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(cache_folder, exist_ok=True)


def cache_func(func, name, *args, **kwargs):
    filename = os.path.join(cache_folder, name)
    if os.path.exists(filename):
        # print(f"{name} already exists")
        with open(filename, "rb") as f:
            result = pickle.load(f)
    else:
        result = func(*args, **kwargs)
        with open(filename, "wb") as f:
            pickle.dump(result, f)
    return result


class Step:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"Step{self.name}"


def runner(func):

    @wraps(func)
    def call(*args, **kwargs):

        signature = inspect.signature(func)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        kwargs = dict(bound_arguments.arguments.items())
        process = func(**kwargs)
        verbose = kwargs.get('verbose', True)
        breakpoint = kwargs.get('breakpoint', False)
        tag = kwargs.get('tag', func.__name__)
        final_result = kwargs.get('final_result', True)
        os.makedirs(os.path.join(cache_folder, tag), exist_ok=True)
        steps = {}
        for num, (name, run) in enumerate(process.items()):
            if not (isinstance(run, list) or isinstance(run, tuple)):
                raise TypeError(f"{run} should be set tuple or list")
            filename = os.path.join(tag, f"{str(name)}.pkl")
            if len(run) == 1:
                steps[str(name)] = run[0]()
            else:
                assert len(run) == 2, f"{run} only with function and params"
                f, p = run
                if isinstance(p, list):
                    for n, v in enumerate(p):
                        if isinstance(v, Step):
                            p[n] = steps[str(v)]
                    steps[str(name)] = cache_func(f, filename, *p)
                elif isinstance(p, dict):
                    for k, v in p.items():
                        if isinstance(v, Step):
                            p[k] = steps[str(v)]
                    steps[str(name)] = cache_func(f, filename, **p)
                else:
                    if isinstance(p, Step): # 결과가 tuple
                        res = steps[str(p)]
                        if isinstance(res, tuple):
                            steps[str(name)] = cache_func(f, filename, *res)
                        else:
                            steps[str(name)] = cache_func(f, filename, res)
                    else:
                        steps[str(name)] = cache_func(f, filename, p)
            if verbose:
                func_name = run[0].__name__
                print(f"<---------------------- {str(name)} : {func_name} ---------------------->\n{steps[str(name)]}\n")
            if breakpoint == str(name):
                return steps[str(name)]
        if final_result:
            return steps[str(name)] # 마지막 값을 반환
        else:
            return steps
    return call
