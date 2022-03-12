#!/usr/bin/env python3
import numpy as np
from dataset.dataset import load_dataset
import torch
import argparse
from benchmark.evaluate import evaluate
from model.Linear import Linear

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', default='checkpoints/bestmodel.pts')
parser.add_argument('--datapath', default='data/dataset_test.bin')
args = parser.parse_args()

def main():
    datasets = load_dataset(args.datapath)['dataset']
    model = Linear()
    with open(args.modelpath, 'rb') as fp:
        model.load_state_dict(torch.load(fp))
    model.eval()
    print("test accuracy: {}".format(np.mean(evaluate(model, datasets))))
    return 0
    
if __name__ == '__main__':
    exit(main())