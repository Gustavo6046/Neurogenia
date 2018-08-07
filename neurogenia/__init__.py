import random        # Common Random
import numpy as np   # Numerical Processing
import time          # Chronological Statistics

try:
    from . import neuros # Neural Networking Library
    
except ImportError:
    import neuros


def evolve_target(model, inputs, outputs, population=10, mutation=0.3, elitism=0.2, survival=0.5, generations=1):
    populus = [model.copy(False, True) for _ in range(max(2, population))]
    t = []
    tl = time.time()
    
    for gen in range(generations):
        leaderboard = sorted(populus, key=lambda x: -x.target_fitness(inputs, outputs))
        
        if gen > 0:
            print("Generation #{:6d} | Best: {:.4f}% | Worst: {:.4f}% | ETA: {:.4f}s".format(gen, (leaderboard[0].target_fitness(inputs, outputs) + 1) * 100, (leaderboard[-1].target_fitness(inputs, outputs) + 1) * 100, sum(t) * (generations - len(t)) / len(t)))
        
        num_surv = max(2, int(survival * population))
        survivors = leaderboard[:num_surv]
        
        while len(survivors) < population:
            a, b = np.random.choice(survivors, (2,))
            res = {}
            
            ag = a.get_genes()
            bg = b.get_genes()
            
            for k, v in ag.items():
                res[k] = random.choice((ag[k], bg[k])) + np.random.uniform(-mutation, mutation, (len(ag[k]),))
                
            c = model.copy(False, True)
            c.set_genes(res)
                
            survivors.append(c)
            
        num_elit = max(0, int(elitism))
        
        for m in survivors[num_elit: num_surv]:
            gn = m.get_genes()
            res = {}
            
            for k, v in gn.items():
                res[k] = v + np.random.uniform(-mutation, mutation, (len(v),))
                
            m.set_genes(res)
            
        populus = survivors
        
        t.append(time.time() - tl)
        tl = time.time()

    leaderboard = sorted(populus, key=lambda x: -x.target_fitness(inputs, outputs))
    print("Results | Best: {:.4f}% | Worst: {:.4f}%".format((leaderboard[0].target_fitness(inputs, outputs) + 1) * 100, (leaderboard[-1].target_fitness(inputs, outputs) + 1) * 100))
        
    return leaderboard[0]

def evolve_custom(model, fitness, population=10, mutation=0.3, elitism=0.2, survival=0.5, generations=1):
    populus = [model.copy(False, True) for _ in range(max(2, population))]
    t = []
    tl = time.time()
    
    for gen in range(generations):
        leaderboard = sorted(populus, key=lambda x: -fitness(x))
        
        if gen > 0:
            print("Generation #{:6d} | Best: {:.4f}% | Worst: {:.4f}% | ETA: {:.4f}s".format(gen, (fitness(leaderboard[0]) + 1) * 100, (fitness(leaderboard[-1]) + 1) * 100, sum(t) * (generations - len(t)) / len(t)))
        
        num_surv = max(2, int(survival * population))
        survivors = leaderboard[:num_surv]
        
        while len(survivors) < population:
            a, b = np.random.choice(survivors, (2,))
            res = {}
            
            ag = a.get_genes()
            bg = b.get_genes()
            
            for k, v in ag.items():
                res[k] = random.choice((ag[k], bg[k])) + np.random.uniform(-mutation, mutation, (len(ag[k]),))
                
            c = model.copy(False, True)
            c.set_genes(res)
                
            survivors.append(c)
            
        num_elit = max(0, int(elitism))
        
        for m in survivors[num_elit: num_surv]:
            gn = m.get_genes()
            res = {}
            
            for k, v in gn.items():
                res[k] = v + np.random.uniform(-mutation, mutation, (len(v),))
                
            m.set_genes(res)
            
        populus = survivors
        
        t.append(time.time() - tl)
        tl = time.time()

    leaderboard = sorted(populus, key=lambda x: -fitness(x))
    print("Results | Best: {:.4f}% | Worst: {:.4f}%".format((leaderboard[0].target_fitness(inputs, outputs) + 1) * 100, (leaderboard[-1].target_fitness(inputs, outputs) + 1) * 100))
        
    return leaderboard[0]