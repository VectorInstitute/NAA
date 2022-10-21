import os
from collections import defaultdict
from itertools import chain
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from utils import *


class GeneralMCDataset(Dataset):
    def __init__(self, args, tokenizer, dataset_path, limit=None):
        """
        Initialize dataset for a general dataset from a json file organized like how described in the README.

        """
        if args.second_loss == "bc":
            raise ValueError(
                "Using binary classification as the second head loss is not supported by the General Dataset class currently"
            )

        print("Loading Dataset")
        # using ijson in case dataset is very large
        with open(dataset_path, "r") as f:
            json_data = json.load(f)

        if isinstance(json_data[0]["context"][0], str):
            raise Exception("Detected string in value, dataset may not be tokenized.")

        print("Building inputs and labels")
        self.input_ids = []
        self.token_type_ids = []
        self.mc_token_ids = []
        self.lm_labels = []
        self.mc_labels = []
        self.start_ids = []
        self.pad_token = tokenizer.convert_tokens_to_ids("<pad>")

        for i in tqdm(range(len(json_data))):
            dialog = json_data[i]

            # could also get distractor to further train with second head
            # just takes reply of next item in dataset as distractor. Kind of lazy, could do better randomized distractors
            # distractor = json_data[(i+1) // len(json_data)]['reply']

            instance = build_input_from_segments(
                dialog["context"],
                dialog["history"],
                dialog["reply"],
                tokenizer,
                lm_labels=True,
            )

            # skip inputs that are too long
            if len(instance["input_ids"]) > args.max_len:
                continue

            self.start_ids.append(instance["start_idx"])
            self.mc_labels.append(1)
            self.input_ids.append([[0], instance["input_ids"]])
            self.token_type_ids.append([[0], instance["token_type_ids"]])
            self.mc_token_ids.append([0, instance["mc_token_ids"]])
            self.lm_labels.append([[-100], instance["lm_labels"]])

    def __len__(self):
        return len(self.mc_labels)

    def __getitem__(self, idx):
        instance = {
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "mc_token_ids": self.mc_token_ids[idx],
            "lm_labels": self.lm_labels[idx],
            "mc_labels": self.mc_labels[idx],
            "reply_start": self.start_ids[idx],
            "pad_token": self.pad_token,
        }
        return instance
