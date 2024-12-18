# Memory Reduced Meta Learning with Guaranteed Convergence

## Introduction to our code

Our code is organized into four sections with the same names as the datasets, each corresponding to the experiments conducted on different datasets as described in the paper. Each section includes implementations of all comparison algorithms on the respective dataset, as well as the recording of relevant metrics, such as `meta-learning accuracy`, `memory costs`, and `wallclock time`.

## Environments for meta-learning experiments

As stated in our paper, our meta-learning experiments were conducted based on the `learn2learn` third-party library, which provides the utilities and unified interface for data preparation and implementations of the MAML and ANIL methods. Additionally, we used the `swanlab` third-party library for logging and recording metrics during the experiments. The following are the key dependency tools and packages, along with their versions, used in our experiments. They are compatible with both Windows and Linux operating systems.

- python=3.8
- torch=1.10.0
- torchvision=0.11.1
- cuda=11.3
- learn2learn=0.1.5
- swanlab=0.3.0

## How to run our code

In each section, the experiments can be launched from the `main` file in an IDE or executed from the command line with parameters. The following are the key parameters and their meanings.

- '-t' or '--test_num': Identifier for different algorithms.
- '-s' or '--seed': Setting for the random seed.
- '-p' or '--pretrain': Whether to load a pretrained model.
- '-i' or '--iterations': Outer-loop iteration number.
- '-n' or '--test_only': Whether to only conduct fine-tuning during the testing phase.

Additionally, other important parameters, such as the size of the task batch and the setup of few-shot tasks (e.g., 'ways' and 'shots'), can be manually adjusted in the `main` file.
