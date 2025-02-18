import argparse
import os
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=False, default="models1")
args = parser.parse_args()

model_list = os.listdir(args.checkpoint)
if len(model_list) > 0:
    last_model_files = os.path.join(args.checkpoint, model_list[-1])
last_model_list = os.listdir(last_model_files)
if len(last_model_list) > 0:
    last_model = os.path.join(last_model_files, last_model_list[-1])

print(last_model)