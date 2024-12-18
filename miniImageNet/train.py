import copy
import math
from typing import List, Dict

import numpy as np
import torch
from torch import nn
import torchvision as tv
import torchvision.transforms as transforms
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import hypergradient as hg
import csv
import time
from optimizier import MAML, ANIL, Algorithm1, ITDBiO
import swanlab
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class DTrainer:
    def __init__(self,
                 dataset="CIFAR-FS",
                 iterations=1000,
                 test_batch_task=32,
                 batch_task=32,
                 ways=5,
                 shots=5,
                 seed=42,
                 fname=None,
                 test_lr=0.05,
                 test_adapt_steps=20,
                 test_iters=20,
                 test_only_flag=False,
                 pretrain_flag=0,
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.iterations = iterations
        self.test_batch_task = test_batch_task
        self.batch_task = batch_task
        self.ways = ways
        self.shots = shots
        self.seed = seed
        self.test_iters = test_iters
        self.test_adapt_lr = test_lr
        self.test_adapt_steps = test_adapt_steps
        self.test_only_flag = test_only_flag
        self.pretrained_flag = pretrain_flag
        if dataset == 'miniImageNet':
            self.hidden_size = 32
            self.head_dim = 25 * self.hidden_size
        else:
            raise ValueError(f'{dataset} is not supported')

        self.fname = fname

        self.train_iterations = []
        self.test_iterations: List = []

        self.train_accuracy = []
        self.train_error = []
        self.valid_accuracy = []
        self.valid_error = []
        self.test_accuracy = []
        self.test_error = []

        self.run_time = []
        self.opt_name = "None"

        self.load_data()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def zero_grad(self, param):
        for p in param:
            if p is not None and p.grad is not None:
                p.grad.zero_()

    def zero_param(self, param):
        for p in param:
            if p is not None:
                p.data.zero_()

    def _save(self):
        with open(self.fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator='\n')
            file.writerow([f"{self.opt_name}, {self.batch_task}, {self.iterations}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_error)
            file.writerow(self.train_accuracy)
            file.writerow(self.valid_error)
            file.writerow(self.valid_accuracy)
            file.writerow([])
            file.writerow(self.test_iterations)
            file.writerow(self.test_error)
            file.writerow(self.test_accuracy)
            file.writerow([])
            file.writerow([])

    def model_save(self, iter=1000, **kwargs):
        print('*' * 20)
        for key, value in kwargs.items():
            torch.save(value,
                       f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}_{key}_{iter}.pth")
        print('*' * 5 + 'Model Saved' + '*' * 5)

    def load_data(self):
        print("==> Loading Data")

        self.train_dataset = None
        self.train_tasks = None
        self.valid_dataset = None
        self.valid_tasks = None
        self.test_dataset = None
        self.test_tasks = None

        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_validation = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

        if self.dataset == "miniImageNet":
            train_dataset = l2l.vision.datasets.MiniImagenet(root='./dataset',
                                                             mode='train',
                                                             download=True,
                                                             )
            valid_dataset = l2l.vision.datasets.MiniImagenet(root='./dataset',
                                                             mode='validation',
                                                             download=True,
                                                             )
            test_dataset = l2l.vision.datasets.MiniImagenet(root='./dataset',
                                                            mode='test',
                                                            download=True,
                                                            )
        else:
            raise ValueError(f'{self.dataset} is not supported')

        self.train_dataset = l2l.data.MetaDataset(train_dataset)
        self.valid_dataset = l2l.data.MetaDataset(valid_dataset)
        self.test_dataset = l2l.data.MetaDataset(test_dataset)

        train_transforms = [
            FusedNWaysKShots(self.train_dataset, n=self.ways, k=2 * self.shots),
            LoadData(self.train_dataset),
            RemapLabels(self.train_dataset),
            ConsecutiveLabels(self.train_dataset),
        ]
        self.train_tasks = l2l.data.TaskDataset(self.train_dataset,
                                                task_transforms=train_transforms,
                                                num_tasks=20000)

        valid_transforms = [
            FusedNWaysKShots(self.valid_dataset, n=self.ways, k=2 * self.shots),
            LoadData(self.valid_dataset),
            ConsecutiveLabels(self.valid_dataset),
            RemapLabels(self.valid_dataset),
        ]
        self.valid_tasks = l2l.data.TaskDataset(self.valid_dataset,
                                                task_transforms=valid_transforms,
                                                num_tasks=600)

        test_transforms = [
            FusedNWaysKShots(self.test_dataset, n=self.ways, k=2 * self.shots),
            LoadData(self.test_dataset),
            RemapLabels(self.test_dataset),
            ConsecutiveLabels(self.test_dataset),
        ]
        self.test_tasks = l2l.data.TaskDataset(self.test_dataset,
                                               task_transforms=test_transforms,
                                               num_tasks=600)

    def top_k_accuracy(self, predictions, targets, k=1):
        _, top_k_preds = predictions.topk(k, dim=1)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        correct = correct.any(dim=1).float().sum()
        return correct / targets.size(0)

    def trainer(self):
        print(
            f"==> Starting Training for {self.opt_name}, {self.iterations} epochs on the {self.dataset} dataset, "
            f"via {self.device}")

    def model_setup(self):
        pass

    def each_train(self, *args, **kwargs):
        pass

    def each_valid(self, *args, **kwargs):
        pass

    def each_test(self, *args, **kwargs):
        pass


def set_swanlab_config(swanlab_config_keys, swanlab_config_values):
    for key, value in zip(swanlab_config_keys, swanlab_config_values):
        swanlab.config.set(key, value)


class MAMLTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = MAML
        self.opt_name = "MAML"
        self.model_setup()
        self.trainer()
        self._save()

    def model_setup(self):
        self.feature = None
        self.optimizer = None
        self.head = None
        self.model = None
        if self.dataset == 'miniImageNet':
            # self.head_dim = 1600
            self.inner_steps = 3
            self.inner_lr = 0.5
            self.outer_lr = 0.005
        else:
            raise ValueError(f'{self.dataset} is not supported')
        self.feature = l2l.vision.models.ConvBase(output_size=self.ways, channels=3, max_pool=True,
                                                  hidden=self.hidden_size,
                                                  )
        self.feature = torch.nn.Sequential(self.feature, Lambda(lambda x: x.view(-1, self.head_dim)))
        self.head = torch.nn.Linear(self.head_dim, self.ways)

        if self.pretrained_flag:
            print("==> Loading Pretrained Model")
            self.feature.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_feature_{self.pretrained_flag}.pth")
            )
            self.head.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_head_{self.pretrained_flag}.pth")
            )

            print("==> Pretrained Model Loaded")

        self.model = torch.nn.Sequential(self.feature, self.head)
        self.model = l2l.algorithms.MAML(self.model, lr=self.inner_lr)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)

        swanlab_config_keys = ["inner_steps", "inner_lr", "outer_lr"]
        swanlab_config_values = [self.inner_steps, self.inner_lr, self.outer_lr]
        set_swanlab_config(swanlab_config_keys, swanlab_config_values)

    def trainer(self):
        super().trainer()
        if not self.test_only_flag:
            for k in range(self.iterations):
                self.each_train(k, self.train_tasks, self.inner_steps)
                self.each_valid(k, self.valid_tasks, self.inner_steps, self.inner_lr)
                if (k + 1) % 1000 == 0:
                    super().model_save(
                        iter=k + 1 if not self.pretrained_flag else (self.pretrained_flag + self.iterations),
                        **{
                            "feature": self.feature.state_dict(),
                            "head": self.head.state_dict()
                        })
        self.each_test(self.test_tasks, adaptation_steps=self.test_adapt_steps, lr=self.test_adapt_lr,
                       test_iteration=self.test_iters)

    def each_train(self, k, train_tasks, adaptation_steps):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        self.optimizer.zero_grad()
        for task in range(self.batch_task):
            train_data, train_labels = train_tasks.sample()
            train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)

            learner = self.model.clone()

            train_adaptation_indices = np.zeros(train_data.size(0), dtype=bool)
            train_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            train_evaluation_indices = torch.from_numpy(~train_adaptation_indices)
            train_adaptation_indices = torch.from_numpy(train_adaptation_indices)
            train_adaptation_data, train_adaptation_labels = train_data[train_adaptation_indices], \
                train_labels[train_adaptation_indices]
            train_evaluation_data, train_evaluation_labels = train_data[train_evaluation_indices], \
                train_labels[train_evaluation_indices]
            for n in range(adaptation_steps):
                train_adaptation_error = self.criterion(learner(train_adaptation_data),
                                                        train_adaptation_labels)
                learner.adapt(train_adaptation_error)

            predictions = learner(train_evaluation_data)
            train_evaluation_error = self.criterion(predictions, train_evaluation_labels)
            train_evaluation_acc = self.top_k_accuracy(predictions, train_evaluation_labels)
            train_evaluation_error.backward()
            meta_train_error += train_evaluation_error.item()
            meta_train_accuracy += train_evaluation_acc.item()
        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / self.batch_task)
        self.optimizer.step()

        average_train_accuracy = meta_train_accuracy / self.batch_task
        average_train_error = meta_train_error / self.batch_task
        self.train_accuracy.append(average_train_accuracy)
        self.train_error.append(average_train_error)

        end_time = time.time()
        self.run_time.append(end_time - start_time)
        print('\n')
        print('Iteration', k)
        print('Meta Train Error', average_train_error)
        print('Meta Train Accuracy', average_train_accuracy)
        print('time per training iteration', end_time - start_time)
        swanlab.log({"Meta Train Error": math.log10(average_train_error),
                     "Meta Train Accuracy": average_train_accuracy,
                     "Peak CUDA Memory": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     })

    def each_valid(self, k, valid_tasks, adaptation_steps, lr):
        valid_error = 0.0
        valid_accuracy = 0.0
        for task in range(self.batch_task):
            learner = self.model.clone()

            batch = valid_tasks.sample()
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)

            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            for step in range(adaptation_steps):
                output = learner(adaptation_data)
                train_error = self.criterion(output, adaptation_labels)
                learner.adapt(train_error)

            for p in self.model.parameters():
                p.requires_grad = True

            predictions = learner(evaluation_data)
            valid_error += self.criterion(predictions, evaluation_labels).item()
            valid_accuracy += self.top_k_accuracy(predictions, evaluation_labels).item()

        average_valid_error = valid_error / self.batch_task
        average_valid_accuracy = valid_accuracy / self.batch_task
        self.valid_error.append(average_valid_error)
        self.valid_accuracy.append(average_valid_accuracy)

        print('Valid Error', average_valid_error)
        print('Valid Accuracy', average_valid_accuracy)
        swanlab.log({"Meta Valid Error": math.log10(average_valid_error),
                     "Meta Valid Accuracy": average_valid_accuracy})

    def each_test(self, test_tasks, adaptation_steps, lr, test_iteration):
        new_feature = copy.deepcopy(self.feature)
        head_global = copy.deepcopy(self.head)
        if self.test_only_flag:
            new_feature.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                           f"_feature_{self.pretrained_flag}.pth"))
            head_global.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                           f"_head_{self.pretrained_flag}.pth"))
        head_global = l2l.algorithms.MAML(head_global, lr=lr)

        for step in range(test_iteration):
            test_error = 0.0
            test_accuracy = 0.0
            self.zero_grad(new_feature.parameters())
            self.zero_grad(head_global.parameters())
            for task in range(self.test_batch_task):
                learner = head_global.clone()
                for p in new_feature.parameters():
                    p.requires_grad = False

                batch = test_tasks.sample()
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
                evaluation_indices = torch.from_numpy(~adaptation_indices)
                adaptation_indices = torch.from_numpy(adaptation_indices)
                adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
                evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

                for i in range(adaptation_steps):
                    feature_output = new_feature(adaptation_data).detach()
                    train_error = self.criterion(learner(feature_output), adaptation_labels)
                    learner.adapt(train_error)

                for p in new_feature.parameters():
                    p.requires_grad = True

                predictions = learner(new_feature(evaluation_data))
                test_error += (self.criterion(predictions, evaluation_labels)).item()
                test_accuracy += (self.top_k_accuracy(predictions, evaluation_labels)).item()

            self.test_iterations.append(step)
            average_test_error = test_error / self.batch_task
            average_test_accuracy = test_accuracy / self.batch_task
            self.test_error.append(average_test_error)
            self.test_accuracy.append(average_test_accuracy)

            print('Test Iteration', step)
            print('Test Error', average_test_error)
            print('Test Accuracy', average_test_accuracy)
            swanlab.log({"Meta Test Error": math.log10(average_test_error),
                         "Meta Test Accuracy": average_test_accuracy})


class ANILTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = ANIL
        self.opt_name = "ANIL"
        self.model_setup()
        self.trainer()
        self._save()

    def model_setup(self):
        self.feature = None
        self.optimizer = None
        self.head = None
        self.all_parameters = None
        if self.dataset == 'miniImageNet':
            # self.head_dim = 1600
            self.inner_steps = 5
            self.fast_lr = 0.1
            self.meta_head_lr = 0.001
            self.meta_feature_lr = 0.001
        else:
            raise ValueError(f'{self.dataset} is not supported')
        self.feature = l2l.vision.models.ConvBase(output_size=self.ways, channels=3, max_pool=True,
                                                  hidden=self.hidden_size,
                                                  )
        self.feature = torch.nn.Sequential(self.feature, Lambda(lambda x: x.view(-1, self.head_dim)))
        self.feature.to(self.device)

        self.head = torch.nn.Linear(self.head_dim, self.ways)
        self.head = l2l.algorithms.MAML(self.head, lr=self.fast_lr)
        self.head.to(self.device)

        if self.pretrained_flag:
            print("==> Loading Pretrained Model")
            inner_steps_str: str = str(self.inner_steps)
            self.feature.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_feature_{self.pretrained_flag}.pth")
            )
            self.head.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_head_{self.pretrained_flag}.pth")
            )
            print("==> Pretrained Model Loaded")

        self.all_parameters = (list(self.feature.parameters()) +
                               list(self.head.parameters()))
        self.optimizer = torch.optim.Adam([
            {'params': list(self.head.parameters()),
             'lr': self.meta_head_lr},
            {'params': list(self.feature.parameters()),
             'lr': self.meta_feature_lr}
        ])
        swanlab_config_keys = ["inner_steps", "fast_lr", "meta_head_lr", "meta_feature_lr"]
        swanlab_config_values = [self.inner_steps, self.fast_lr, self.meta_head_lr, self.meta_feature_lr]
        set_swanlab_config(swanlab_config_keys, swanlab_config_values)

    def trainer(self):
        super().trainer()
        if not self.test_only_flag:
            for k in range(self.iterations):
                self.each_train(k, self.train_tasks, self.inner_steps)
                self.each_valid(k, self.valid_tasks, self.inner_steps, self.fast_lr)
                if (k + 1) % 1000 == 0:
                    super().model_save(
                        iter=k + 1 if not self.pretrained_flag else (self.pretrained_flag + self.iterations),
                        **{
                            "feature": self.feature.state_dict(),
                            "head": self.head.state_dict()
                        })
        self.each_test(self.test_tasks, adaptation_steps=self.test_adapt_steps, lr=self.test_adapt_lr,
                       test_iteration=self.test_iters)

    def each_train(self, k, train_tasks, adaptation_steps):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        self.optimizer.zero_grad()
        for task in range(self.batch_task):
            train_data, train_labels = train_tasks.sample()
            train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)

            learner = self.head.clone()

            train_adaptation_indices = np.zeros(train_data.size(0), dtype=bool)
            train_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            train_evaluation_indices = torch.from_numpy(~train_adaptation_indices)
            train_adaptation_indices = torch.from_numpy(train_adaptation_indices)
            train_adaptation_data, train_adaptation_labels = (train_data[train_adaptation_indices],
                                                              train_labels[train_adaptation_indices])
            train_evaluation_data, train_evaluation_labels = (train_data[train_evaluation_indices],
                                                              train_labels[train_evaluation_indices])

            for n in range(adaptation_steps):
                train_adaptation_error = self.criterion(learner(self.feature(train_adaptation_data)),
                                                        train_adaptation_labels)
                learner.adapt(train_adaptation_error)
            predictions = learner(self.feature(train_evaluation_data))
            train_evaluation_error = self.criterion(predictions, train_evaluation_labels)
            train_evaluation_acc = self.top_k_accuracy(predictions, train_evaluation_labels)
            train_evaluation_error.backward()
            meta_train_error += train_evaluation_error.item()
            meta_train_accuracy += train_evaluation_acc.item()

        for p in self.all_parameters:
            p.grad.data.mul_(1.0 / self.batch_task)
        self.optimizer.step()

        average_train_accuracy = meta_train_accuracy / self.batch_task
        average_train_error = meta_train_error / self.batch_task
        self.train_accuracy.append(average_train_accuracy)
        self.train_error.append(average_train_error)

        end_time = time.time()
        self.run_time.append(end_time - start_time)
        print('\n')
        print('Iteration', k)
        print('Meta Train Error', average_train_error)
        print('Meta Train Accuracy', average_train_accuracy)
        print('time per training iteration', end_time - start_time)
        swanlab.log({"Meta Train Error": math.log10(average_train_error),
                     "Meta Train Accuracy": average_train_accuracy,
                     "Peak CUDA Memory": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     })

    def each_valid(self, k, valid_tasks, adaptation_steps, lr):
        valid_error = 0.0
        valid_accuracy = 0.0
        for task in range(self.batch_task):
            learner = self.head.clone()
            for p in self.feature.parameters():
                p.requires_grad = False
            batch = valid_tasks.sample()
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)

            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            for step in range(adaptation_steps):
                feature_output = self.feature(adaptation_data).detach()
                train_error = self.criterion(learner(feature_output), adaptation_labels)
                learner.adapt(train_error)

            for p in self.feature.parameters():
                p.requires_grad = True

            predictions = learner(self.feature(evaluation_data))
            valid_error += self.criterion(predictions, evaluation_labels).item()
            valid_accuracy += self.top_k_accuracy(predictions, evaluation_labels).item()

        average_valid_error = valid_error / self.batch_task
        average_valid_accuracy = valid_accuracy / self.batch_task
        self.valid_error.append(average_valid_error)
        self.valid_accuracy.append(average_valid_accuracy)

        print('Valid Error', average_valid_error)
        print('Valid Accuracy', average_valid_accuracy)
        swanlab.log({"Meta Valid Error": math.log10(average_valid_error),
                     "Meta Valid Accuracy": average_valid_accuracy})

    def each_test(self, test_tasks, adaptation_steps, lr, test_iteration):
        new_feature = copy.deepcopy(self.feature)
        head_global = copy.deepcopy(self.head)
        if self.test_only_flag:
            head_global.lr = lr
            new_feature.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                           f"_feature_{self.pretrained_flag}.pth"))
            head_global.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}"
                           f"_head_{self.pretrained_flag}.pth"))
        for step in range(test_iteration):
            test_error = 0.0
            test_accuracy = 0.0
            self.zero_grad(new_feature.parameters())
            self.zero_grad(head_global.parameters())
            for task in range(self.test_batch_task):
                learner = head_global.clone()
                for p in new_feature.parameters():
                    p.requires_grad = False

                batch = test_tasks.sample()
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
                evaluation_indices = torch.from_numpy(~adaptation_indices)
                adaptation_indices = torch.from_numpy(adaptation_indices)
                adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
                evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

                for i in range(adaptation_steps):
                    feature_output = new_feature(adaptation_data).detach()
                    train_error = self.criterion(learner(feature_output), adaptation_labels)
                    learner.adapt(train_error)

                for p in new_feature.parameters():
                    p.requires_grad = True

                predictions = learner(new_feature(evaluation_data))
                test_error += (self.criterion(predictions, evaluation_labels)).item()
                test_accuracy += (self.top_k_accuracy(predictions, evaluation_labels)).item()

            self.test_iterations.append(step)
            average_test_error = test_error / self.batch_task
            average_test_accuracy = test_accuracy / self.batch_task
            self.test_error.append(average_test_error)
            self.test_accuracy.append(average_test_accuracy)

            print('Test Iteration', step)
            print('Test Error', average_test_error)
            print('Test Accuracy', average_test_accuracy)
            swanlab.log({"Meta Test Error": math.log10(average_test_error),
                         "Meta Test Accuracy": average_test_accuracy})


class Algorithm1Trainer(DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = Algorithm1
        self.opt_name = "Algorithm 1"
        self.model_setup()
        self.trainer()
        super()._save()

    def model_setup(self):
        self.feature = None
        self.optimizer = None
        self.head = None
        self.feature_parameters = None

        if self.dataset == 'miniImageNet':
            # self.head_dim = 1600
            self.inner_steps = 20
            self.fast_lr = 0.05
            self.meta_lr = 0.001
        else:
            raise ValueError(f'{self.dataset} is not supported')
        self.CG_steps = 20

        self.feature = l2l.vision.models.ConvBase(output_size=self.ways, channels=3, max_pool=True,
                                                  hidden=self.hidden_size,
                                                  )
        self.feature = torch.nn.Sequential(self.feature, Lambda(lambda x: x.view(-1, self.head_dim)))
        self.feature.to(self.device)
        self.head = torch.nn.Linear(self.head_dim, self.ways)
        self.head = l2l.algorithms.MAML(self.head, lr=self.fast_lr)
        self.head.to(self.device)

        self.previous_head_list = []
        self.previous_v_list = []
        for i in range(self.batch_task):
            self.previous_head_list.append(self.head.state_dict())

        if self.pretrained_flag:
            print("==> Loading Pretrained Model")
            inner_steps_str: str = str(self.inner_steps)
            meta_lr_str: str = str(self.meta_lr)[2:]
            post_fix = f"{inner_steps_str}_lr{meta_lr_str}"
            self.feature.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/{post_fix}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_feature_{post_fix}_{self.pretrained_flag}.pth")
            )
            self.head.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/{post_fix}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_head_{post_fix}_{self.pretrained_flag}.pth")
            )
            print("==> Pretrained Model Loaded")

        self.feature_parameters = list(self.feature.parameters())
        self.optimizer = torch.optim.Adam(self.feature_parameters, lr=self.meta_lr)
        self.y_optimizer = self.opt(
            params=list(self.head.parameters()),
            name=self.opt_name,
            device=self.device
        )
        swanlab_config_keys = ["inner_steps", "fast_lr", "meta_lr", "CG_steps"]
        swanlab_config_values = [self.inner_steps, self.fast_lr, self.meta_lr, self.CG_steps]
        set_swanlab_config(swanlab_config_keys, swanlab_config_values)

    def re_initialize_y(self, head, previous_head_state):
        with torch.no_grad():
            head.load_state_dict(previous_head_state)
        self.zero_grad(head.parameters())

    def model_save(self, iter=1000, **kwargs):
        print('*' * 20)
        for key, value in kwargs.items():
            torch.save(value,
                       f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}_{key}_"
                       f"{self.inner_steps}_lr{str(self.meta_lr)[2:]}_{iter}.pth")
        print('*' * 5 + 'Model Saved' + '*' * 5)

    def trainer(self):
        super().trainer()
        if not self.test_only_flag:
            for k in range(self.iterations):
                self.each_train(k, self.train_tasks, self.inner_steps)
                self.each_valid(k, self.valid_tasks, self.inner_steps, self.fast_lr)
                if (k + 1) % 1000 == 0:
                    self.model_save(
                        iter=k + 1 if not self.pretrained_flag else (self.pretrained_flag + self.iterations),
                        **{
                            "feature": self.feature.state_dict(),
                            "head": self.head.state_dict()
                        }
                    )
        self.each_test(self.test_tasks, adaptation_steps=self.test_adapt_steps, lr=self.test_adapt_lr,
                       test_iteration=self.test_iters)

    def each_train(self, k, train_tasks, adaptation_steps):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        self.optimizer.zero_grad()
        for task in range(self.batch_task):
            train_data, train_labels = train_tasks.sample()
            train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
            train_adaptation_indices = np.zeros(train_data.size(0), dtype=bool)
            train_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            train_evaluation_indices = torch.from_numpy(~train_adaptation_indices)
            train_adaptation_indices = torch.from_numpy(train_adaptation_indices)
            train_adaptation_data, train_adaptation_labels = train_data[train_adaptation_indices], \
                train_labels[train_adaptation_indices]
            train_evaluation_data, train_evaluation_labels = train_data[train_evaluation_indices], \
                train_labels[train_evaluation_indices]

            learner = self.head.clone()
            for p in self.feature.parameters():
                p.requires_grad = False

            for n in range(adaptation_steps):
                feature_output = self.feature(train_adaptation_data).detach()
                l2_reg = 0.0
                for p in learner.parameters():
                    l2_reg += p.norm(2)
                train_adaptation_error = self.criterion(learner(feature_output),
                                                        train_adaptation_labels)
                learner.adapt(train_adaptation_error)

            self.previous_head_list[task] = copy.deepcopy(learner.state_dict())

            for p in self.feature.parameters():
                p.requires_grad = True
            last_adaptation_predictions = learner(self.feature(train_adaptation_data))
            last_adaptation_error = self.criterion(last_adaptation_predictions, train_adaptation_labels)

            predictions = learner(self.feature(train_evaluation_data))
            train_evaluation_error = self.criterion(predictions, train_evaluation_labels)
            train_evaluation_acc = self.top_k_accuracy(predictions, train_evaluation_labels)

            new_v = hg.CG_Centralized(
                y=list(learner.parameters()),
                x=list(self.feature.parameters()),
                CG_steps=self.CG_steps,
                g_loss=last_adaptation_error,
                f_loss=train_evaluation_error,
            )

            meta_train_error += train_evaluation_error.item()
            meta_train_accuracy += train_evaluation_acc.item()

        for p in self.feature_parameters:
            p.grad.data.mul_(1.0 / self.batch_task)
        self.optimizer.step()

        average_train_accuracy = meta_train_accuracy / self.batch_task
        average_train_error = meta_train_error / self.batch_task
        self.train_accuracy.append(average_train_accuracy)
        self.train_error.append(average_train_error)

        end_time = time.time()
        self.run_time.append(end_time - start_time)
        temp_norm = list()
        for p in self.feature_parameters:
            temp_norm.append((torch.norm(p.grad.data)).to('cpu').numpy())
        hypergradient_norm = np.linalg.norm(temp_norm)

        print('\n')
        print('Iteration', k)
        print('Meta Train Error', average_train_error)
        print('Meta Train Accuracy', average_train_accuracy)
        print('Hypergradient Norm', hypergradient_norm)
        print('time per training iteration', end_time - start_time)
        swanlab.log({"Meta Train Error": math.log10(average_train_error),
                     "Meta Train Accuracy": average_train_accuracy,
                     "Hypergradient Norm": math.log10(hypergradient_norm),
                     "Peak CUDA Memory": torch.cuda.max_memory_allocated() / 1024 / 1024,
                     })

    def each_valid(self, k, valid_tasks, adaptation_steps, lr):
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(self.batch_task):
            learner = self.head.clone()
            for p in self.feature.parameters():
                p.requires_grad = False

            valid_data, valid_labels = valid_tasks.sample()
            valid_data, valid_labels = valid_data.to(self.device), valid_labels.to(self.device)

            valid_adaptation_indices = np.zeros(valid_data.size(0), dtype=bool)
            valid_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            valid_evaluation_indices = torch.from_numpy(~valid_adaptation_indices)
            valid_adaptation_indices = torch.from_numpy(valid_adaptation_indices)
            valid_adaptation_data, valid_adaptation_labels = valid_data[valid_adaptation_indices], \
                valid_labels[valid_adaptation_indices]
            valid_evaluation_data, valid_evaluation_labels = valid_data[valid_evaluation_indices], \
                valid_labels[valid_evaluation_indices]

            for n in range(adaptation_steps):
                feature_output = self.feature(valid_adaptation_data).detach()
                adaptation_error = self.criterion(learner(feature_output), valid_adaptation_labels)
                learner.adapt(adaptation_error)

            for p in self.feature.parameters():
                p.requires_grad = True

            predictions = learner(self.feature(valid_evaluation_data))
            valid_evaluation_error = self.criterion(predictions, valid_evaluation_labels)
            valid_evaluation_acc = self.top_k_accuracy(predictions, valid_evaluation_labels,
                                                       )

            meta_valid_error += valid_evaluation_error.item()
            meta_valid_accuracy += valid_evaluation_acc.item()

        average_valid_accuracy = meta_valid_accuracy / self.batch_task
        average_valid_error = meta_valid_error / self.batch_task
        self.valid_accuracy.append(average_valid_accuracy)
        self.valid_error.append(average_valid_error)

        print('Meta Valid Error', average_valid_error)
        print('Meta Valid Accuracy', average_valid_accuracy)
        swanlab.log({"Meta Valid Error": math.log10(average_valid_error),
                     "Meta Valid Accuracy": average_valid_accuracy})

    def each_test(self, test_tasks, adaptation_steps, lr, test_iteration):
        new_feature = copy.deepcopy(self.feature)
        head_global = copy.deepcopy(self.head)
        head_global.lr = lr
        if self.test_only_flag:
            lr_str: str = str(self.meta_lr)[2:]
            new_feature.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/{self.inner_steps}_lr{lr_str}/"
                           f"seed_{self.seed}_algorithm_{self.opt_name}_feature_{self.inner_steps}_lr{lr_str}"
                           f"_{self.pretrained_flag}.pth")
            )
            head_global.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/{self.inner_steps}_lr{lr_str}/"
                           f"seed_{self.seed}_algorithm_{self.opt_name}_head_{self.inner_steps}_lr{lr_str}"
                           f"_{self.pretrained_flag}.pth")
            )
        for step in range(test_iteration):
            meta_test_error = 0.0
            meta_test_accuracy = 0.0
            self.zero_grad(new_feature.parameters())
            self.zero_grad(head_global.parameters())
            for task in range(self.test_batch_task):
                learner = head_global.clone()
                for p in new_feature.parameters():
                    p.requires_grad = False

                test_data, test_labels = test_tasks.sample()
                test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)

                test_adaptation_indices = np.zeros(test_data.size(0), dtype=bool)
                test_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
                test_evaluation_indices = torch.from_numpy(~test_adaptation_indices)
                test_adaptation_indices = torch.from_numpy(test_adaptation_indices)
                test_adaptation_data, test_adaptation_labels = (test_data[test_adaptation_indices],
                                                                test_labels[test_adaptation_indices])
                test_evaluation_data, test_evaluation_labels = (test_data[test_evaluation_indices],
                                                                test_labels[test_evaluation_indices])
                for n in range(adaptation_steps):
                    feature_output = new_feature(test_adaptation_data).detach()
                    adaptation_error = self.criterion(learner(feature_output), test_adaptation_labels)
                    learner.adapt(adaptation_error)

                for p in new_feature.parameters():
                    p.requires_grad = True
                predictions = learner(new_feature(test_evaluation_data))
                test_evaluation_error = self.criterion(predictions, test_evaluation_labels)
                test_evaluation_acc = self.top_k_accuracy(predictions, test_evaluation_labels)

                meta_test_error += test_evaluation_error.item()
                meta_test_accuracy += test_evaluation_acc.item()

            self.test_iterations.append(step)
            average_test_error = meta_test_error / self.batch_task
            average_test_accuracy = meta_test_accuracy / self.batch_task
            self.test_error.append(average_test_error)
            self.test_accuracy.append(average_test_accuracy)

            print('Test Iteration', step)
            print('Test Error', average_test_error)
            print('Test Accuracy', average_test_accuracy)
            swanlab.log({"Meta Test Error": math.log10(average_test_error),
                         "Meta Test Accuracy": average_test_accuracy})


class ITDBiOTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = ITDBiO
        self.opt_name = "ITD-BiO"
        self.model_setup()
        self.trainer()
        self._save()

    def model_setup(self):
        self.feature = None
        self.optimizer = None
        self.head = None
        self.feature_parameters = None
        if self.dataset == 'miniImageNet':
            # self.head_dim = 1600
            self.inner_steps = 20
            self.fast_lr = 0.05
            self.meta_lr = 0.0001
        else:
            raise ValueError(f'{self.dataset} is not supported')

        self.feature = l2l.vision.models.ConvBase(output_size=self.ways, channels=3, max_pool=True,
                                                  hidden=self.hidden_size,
                                                  )
        self.feature = torch.nn.Sequential(self.feature, Lambda(lambda x: x.view(-1, self.head_dim)))
        self.feature.to(self.device)

        self.head = torch.nn.Linear(self.head_dim, self.ways)
        self.head = l2l.algorithms.MAML(self.head, lr=self.fast_lr)
        self.head.to(self.device)

        if self.pretrained_flag:
            print("==> Loading Pretrained Model")
            inner_steps_str: str = str(self.inner_steps)
            meta_lr_str: str = str(self.meta_lr)[2:]
            post_fix = f"{inner_steps_str}_lr{meta_lr_str}"
            self.feature.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/{post_fix}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_feature_{post_fix}_{self.pretrained_flag}.pth")
            )
            self.head.load_state_dict(
                torch.load(
                    f"./model_params/{self.dataset}/{post_fix}/seed_{self.seed}_algorithm_{self.opt_name}"
                    f"_head_{post_fix}_{self.pretrained_flag}.pth")
            )
            print("==> Pretrained Model Loaded")

        self.feature_parameters = list(self.feature.parameters())
        self.optimizer = torch.optim.Adam(self.feature_parameters, lr=self.meta_lr)

        swanlab_config_keys = ["inner_steps", "fast_lr", "meta_lr"]
        swanlab_config_values = [self.inner_steps, self.fast_lr, self.meta_lr]
        set_swanlab_config(swanlab_config_keys, swanlab_config_values)

    def model_save(self, iter=1000, **kwargs):
        print('*' * 20)
        for key, value in kwargs.items():
            torch.save(value,
                       f"./model_params/{self.dataset}/seed_{self.seed}_algorithm_{self.opt_name}_{key}_"
                       f"{self.inner_steps}_lr{str(self.meta_lr)[2:]}_{iter}.pth")
        print('*' * 5 + 'Model Saved' + '*' * 5)

    def trainer(self):
        super().trainer()
        if not self.test_only_flag:
            for k in range(self.iterations):
                self.each_train(k, self.train_tasks, self.inner_steps)
                self.each_valid(k, self.valid_tasks, self.inner_steps, self.fast_lr)
                if (k + 1) % 1000 == 0:
                    self.model_save(
                        iter=k + 1 if not self.pretrained_flag else (self.pretrained_flag + self.iterations),
                        **{
                            "feature": self.feature.state_dict(),
                            "head": self.head.state_dict()
                        })
        self.each_test(self.test_tasks, adaptation_steps=self.test_adapt_steps, lr=self.test_adapt_lr,
                       test_iteration=self.test_iters)

    def each_train(self, k, train_tasks, adaptation_steps):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        self.optimizer.zero_grad()
        for task in range(self.batch_task):
            train_data, train_labels = train_tasks.sample()
            train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)

            train_adaptation_indices = np.zeros(train_data.size(0), dtype=bool)
            train_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            train_evaluation_indices = torch.from_numpy(~train_adaptation_indices)
            train_adaptation_indices = torch.from_numpy(train_adaptation_indices)
            train_adaptation_data, train_adaptation_labels = train_data[train_adaptation_indices], \
                train_labels[train_adaptation_indices]
            train_evaluation_data, train_evaluation_labels = train_data[train_evaluation_indices], \
                train_labels[train_evaluation_indices]

            learner = self.head.clone()
            for n in range(adaptation_steps):
                l2_reg = 0.0
                for p in learner.parameters():
                    l2_reg += p.norm(2)
                train_adaptation_error = self.criterion(learner(self.feature(train_adaptation_data)),
                                                        train_adaptation_labels) + 0.0001 * l2_reg
                learner.adapt(train_adaptation_error)

            predictions = learner(self.feature(train_evaluation_data))
            train_evaluation_error = self.criterion(predictions, train_evaluation_labels)
            train_evaluation_acc = self.top_k_accuracy(predictions, train_evaluation_labels)

            train_evaluation_error.backward()

            meta_train_error += train_evaluation_error.item()
            meta_train_accuracy += train_evaluation_acc.item()

        for p in self.feature_parameters:
            p.grad.data.mul_(1.0 / self.batch_task)
        self.optimizer.step()

        average_train_accuracy = meta_train_accuracy / self.batch_task
        average_train_error = meta_train_error / self.batch_task
        self.train_accuracy.append(average_train_accuracy)
        self.train_error.append(average_train_error)

        end_time = time.time()
        self.run_time.append(end_time - start_time)

        temp_norm = list()
        for p in self.feature_parameters:
            temp_norm.append((torch.norm(p.grad.data)).to('cpu').numpy())
        hypergradient_norm = np.linalg.norm(temp_norm)

        print('\n')
        print('Iteration', k)
        print('Meta Train Error', average_train_error)
        print('Meta Train Accuracy', average_train_accuracy)
        print('Hypergradient Norm', hypergradient_norm)
        print('time per training iteration', end_time - start_time)
        swanlab.log({"Meta Train Error": math.log10(average_train_error),
                     "Meta Train Accuracy": average_train_accuracy,
                     "Hypergradient Norm": math.log10(hypergradient_norm),
                     "Peak CUDA Memory": torch.cuda.max_memory_allocated() / 1024 / 1024,
                     })

    def each_valid(self, k, valid_tasks, adaptation_steps, lr):
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(self.batch_task):
            learner = self.head.clone()
            for p in self.feature.parameters():
                p.requires_grad = False

            valid_data, valid_labels = valid_tasks.sample()
            valid_data, valid_labels = valid_data.to(self.device), valid_labels.to(self.device)

            valid_adaptation_indices = np.zeros(valid_data.size(0), dtype=bool)
            valid_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
            valid_evaluation_indices = torch.from_numpy(~valid_adaptation_indices)
            valid_adaptation_indices = torch.from_numpy(valid_adaptation_indices)
            valid_adaptation_data, valid_adaptation_labels = valid_data[valid_adaptation_indices], \
                valid_labels[valid_adaptation_indices]
            valid_evaluation_data, valid_evaluation_labels = valid_data[valid_evaluation_indices], \
                valid_labels[valid_evaluation_indices]

            for step in range(adaptation_steps):
                feature_output = self.feature(valid_adaptation_data).detach()
                adaptation_error = self.criterion(learner(feature_output), valid_adaptation_labels)
                learner.adapt(adaptation_error)

            for p in self.feature.parameters():
                p.requires_grad = True

            predictions = learner(self.feature(valid_evaluation_data))
            valid_evaluation_error = self.criterion(predictions, valid_evaluation_labels)
            valid_evaluation_acc = self.top_k_accuracy(predictions, valid_evaluation_labels)

            meta_valid_error += valid_evaluation_error.item()
            meta_valid_accuracy += valid_evaluation_acc.item()

        average_valid_error = meta_valid_error / self.batch_task
        average_valid_accuracy = meta_valid_accuracy / self.batch_task
        self.valid_error.append(average_valid_error)
        self.valid_accuracy.append(average_valid_accuracy)

        print('Meta Valid Error', average_valid_error)
        print('Meta Valid Accuracy', average_valid_accuracy)
        swanlab.log({"Meta Valid Error": math.log10(average_valid_error),
                     "Meta Valid Accuracy": average_valid_accuracy})

    def each_test(self, test_tasks, adaptation_steps, lr, test_iteration):
        new_feature = copy.deepcopy(self.feature)
        head_global = copy.deepcopy(self.head)
        head_global.lr = lr
        if self.test_only_flag:
            lr_str: str = str(self.meta_lr)[2:]
            new_feature.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/{self.inner_steps}_lr{lr_str}/"
                           f"seed_{self.seed}_algorithm_{self.opt_name}_feature_{self.inner_steps}_lr{lr_str}_"
                           f"{self.pretrained_flag}.pth"))
            head_global.load_state_dict(
                torch.load(f"./model_params/{self.dataset}/{self.inner_steps}_lr{lr_str}/"
                           f"seed_{self.seed}_algorithm_{self.opt_name}_head_{self.inner_steps}_lr{lr_str}_"
                           f"{self.pretrained_flag}.pth")
            )
        for step in range(test_iteration):
            meta_test_error = 0.0
            meta_test_accuracy = 0.0
            self.zero_grad(new_feature.parameters())
            self.zero_grad(head_global.parameters())
            for task in range(self.test_batch_task):
                learner = head_global.clone()
                for p in new_feature.parameters():
                    p.requires_grad = False

                test_data, test_labels = test_tasks.sample()
                test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)

                test_adaptation_indices = np.zeros(test_data.size(0), dtype=bool)
                test_adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
                test_evaluation_indices = torch.from_numpy(~test_adaptation_indices)
                test_adaptation_indices = torch.from_numpy(test_adaptation_indices)
                test_adaptation_data, test_adaptation_labels = (test_data[test_adaptation_indices],
                                                                test_labels[test_adaptation_indices])
                test_evaluation_data, test_evaluation_labels = (test_data[test_evaluation_indices],
                                                                test_labels[test_evaluation_indices])
                for n in range(adaptation_steps):
                    feature_output = new_feature(test_adaptation_data).detach()
                    adaptation_error = self.criterion(learner(feature_output), test_adaptation_labels)
                    learner.adapt(adaptation_error)

                for p in new_feature.parameters():
                    p.requires_grad = True

                predictions = learner(new_feature(test_evaluation_data))
                test_evaluation_error = self.criterion(predictions, test_evaluation_labels)
                test_evaluation_acc = self.top_k_accuracy(predictions, test_evaluation_labels)

                meta_test_error += test_evaluation_error.item()
                meta_test_accuracy += test_evaluation_acc.item()

            self.test_iterations.append(step)
            average_test_error = meta_test_error / self.batch_task
            average_test_accuracy = meta_test_accuracy / self.batch_task
            self.test_error.append(average_test_error)
            self.test_accuracy.append(average_test_accuracy)

            print('Test Iteration', step)
            print('Test Error', average_test_error)
            print('Test Accuracy', average_test_accuracy)
            swanlab.log({"Meta Test Error": math.log10(average_test_error),
                         "Meta Test Accuracy": average_test_accuracy})