import os
import argparse
import random
import numpy as np
import torch
from train import MAMLTrainer, ANILTrainer, Algorithm1Trainer, ITDBiOTrainer
import swanlab

batch_task = 32
test_batch_task = 32
test_iter = 20
default_test_lr = 0.05
default_test_steps = 20
ways = 5
shots = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=2, type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-p", "--pretrain", default=0, type=int)
    parser.add_argument("-i", "--iterations", default=1000, type=int)
    parser.add_argument("-n", "--test_only", default=0, type=int)
    return parser.parse_args()


args = parse_args()
seed = args.seed
dataset = 'miniImageNet'
default_pretrain_flag: int = args.pretrain if args.pretrain in (1000, 2000) else 0
default_test_only_flag: bool = True if args.test_only > 0 else False
iterations = args.iterations


cwd = os.getcwd()
results_path = os.path.join(cwd, f"results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

f_name = os.path.join(results_path, f"{dataset}_{args.test_num}_iter{iterations}_{seed}_{ways}_{shots}.csv")

print(f"Test Num {args.test_num}, {f_name}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

Algorithm_name = None
if args.test_num == 0:
    Algorithm_name = "MAML"
elif args.test_num == 1:
    Algorithm_name = "ANIL"
elif args.test_num == 2:
    Algorithm_name = "Algorithm 1"
elif args.test_num == 3:
    Algorithm_name = "ITD-BiO"

swanlab.init(
    experiment_name=f"{Algorithm_name}-{dataset}",
    config={
        "seed": seed,
        "batch": batch_task,
        "Iterations": iterations,
        "ways": ways,
        "shots": shots,
        "test_batch": test_batch_task,
        "test_iter": test_iter,
        "test_lr": default_test_lr,
        "test_steps": default_test_steps,
    },
    logdir="./logs",
    cloud=False
)
if args.test_num == 0:
    MAMLTrainer(dataset=dataset, iterations=iterations, test_batch_task=test_batch_task, batch_task=batch_task,
                ways=ways, shots=shots, seed=seed, test_lr=default_test_lr, test_adapt_steps=default_test_steps,
                test_iters=test_iter, fname=f_name,
                test_only_flag=default_test_only_flag, pretrain_flag=default_pretrain_flag)
elif args.test_num == 1:
    ANILTrainer(dataset=dataset, iterations=iterations, test_batch_task=test_batch_task, batch_task=batch_task,
                ways=ways, shots=shots, seed=seed, test_lr=default_test_lr, test_adapt_steps=default_test_steps,
                test_iters=test_iter, fname=f_name,
                test_only_flag=default_test_only_flag, pretrain_flag=default_pretrain_flag)
elif args.test_num == 2:
    Algorithm1Trainer(dataset=dataset, iterations=iterations, test_batch_task=test_batch_task, batch_task=batch_task,
                      ways=ways, shots=shots, seed=seed, test_lr=default_test_lr, test_adapt_steps=default_test_steps,
                      test_iters=test_iter, fname=f_name,
                      test_only_flag=default_test_only_flag, pretrain_flag=default_pretrain_flag)
elif args.test_num == 3:
    ITDBiOTrainer(dataset=dataset, iterations=iterations, test_batch_task=test_batch_task, batch_task=batch_task,
                  ways=ways, shots=shots, seed=seed, test_lr=default_test_lr, test_adapt_steps=default_test_steps,
                  test_iters=test_iter, fname=f_name,
                  test_only_flag=default_test_only_flag, pretrain_flag=default_pretrain_flag)
else:
    raise ValueError("Invalid test number")
