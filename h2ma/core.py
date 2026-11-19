import numpy as np
from scipy.optimize import minimize
import sys
import itertools
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from h2ma.hypervolume import HyperVolume

def create_region(X:list[np.ndarray], obj_funcs:list[callable])->np.ndarray:
    values = np.array([[obj_func(x) for obj_func in obj_funcs] for x in X])
    values_min = np.min(values, axis=0)
    values_max = np.max(values, axis=0)

    volume = np.prod(values_max - values_min)

    if volume == 0: raise ValueError("The region has zero volume.")
    region = {"X" : X, "mid" : np.mean(values, axis=0), "V" : volume}
    
    return region


def create_initial_region(obj_funcs:list[callable], bounds:np.ndarray)->np.ndarray:
    x_init = np.mean(bounds, axis=0)

    X = []
    for obj_func in obj_funcs:
        res = minimize(obj_func, x_init, method='L-BFGS-B', bounds = bounds)
        X.append(res.x)
    X = np.array(X)
    R = create_region(X, obj_funcs)

    return [R,], X


def x_is_dominated_by_X(x:np.ndarray, X:np.ndarray)->bool:
    nds = NonDominatedSorting().do(np.vstack([X, x]), only_non_dominated_front=True)
    if len(X) in nds:
        return False
    return True


def explore_deterministic(obj_funcs:list[callable], bounds:np.ndarray, R:dict, X:np.ndarray)->np.ndarray|None:
    x0 = np.mean(bounds, axis=0)

    def _objective(x:np.ndarray)->float:
        return np.sum([(obj_func(x) - R["mid"][i])**2 for i, obj_func in enumerate(obj_funcs)])

    res = minimize(_objective, x0, method='L-BFGS-B', bounds = bounds)
    x = res.x
    front_idx = NonDominatedSorting().do(np.vstack([X, x]), only_non_dominated_front=True)
    if len(X) in front_idx:
        return x
    else:
        return None

def exploit(obj_funcs:list[callable], bounds:np.ndarray, x0:np.ndarray, X:np.ndarray, hyper_volume:HyperVolume)->np.ndarray:
    def _objective(x:np.ndarray)->float:
        values_X = np.array([[obj_func(xi) for obj_func in obj_funcs] for xi in X])
        values_x = np.array([obj_func(x) for obj_func in obj_funcs])
        return -hyper_volume.do(np.vstack([values_X, values_x]))
    res = minimize(_objective, x0, method='L-BFGS-B', bounds = bounds)
    
    return res.x

def create_regions(R:dict, x:np.ndarray, obj_funcs:list[callable])->list[dict]:
    new_regions = []

    for XX in itertools.combinations(R["X"], len(R["X"]) - 1):
        RR = create_region(np.vstack([XX, x]), obj_funcs)
        new_regions.append(RR)
    
    return new_regions



def run(obj_funcs:list[callable],  nadir:np.ndarray, x_min:np.ndarray|None = None, x_max:np.ndarray|None = None, dim:int|None = None, max_itr:int = 1000)->np.ndarray:
    hyper_volume = HyperVolume(nadir)
    
    if x_min is None:
        x_min = np.zeros(dim)
    if x_max is None:
        x_max = np.ones(dim)
    bounds = np.stack([x_min, x_max], axis=1)

    Regions, X = create_initial_region(obj_funcs, bounds = bounds)

    itr = 0
    while itr < max_itr and len(Regions) > 0:
        itr += 1
        volumes = [R["V"] for R in Regions]
        idx_max = np.argmax(volumes)
        R = Regions.pop(idx_max)

        x0 = explore_deterministic(obj_funcs, bounds, R, X)
        if x0 is None: continue

        x = exploit(obj_funcs, bounds, x0, X, hyper_volume)
        new_regions = create_regions(R, x, obj_funcs)
        Regions.extend(new_regions)
        X = np.vstack([X, x])

    return X