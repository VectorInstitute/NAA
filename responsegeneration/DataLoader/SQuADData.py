import random
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import *
from DataLoader.dataset_utils import *


class SQuADDataset(Dataset):
    def __init__(self, args, tokenizer, dataset_path):
        """
        Initialize dataset for SQuAD dataset.
        """
        if args.second_loss == "bc":
            raise ValueError(
                "Using binary classification as the second head loss is not supported by the SQuAD Dataset class currently"
            )

        squad_data = get_tokenized_dataset(tokenizer, dataset_path)

        print("Building inputs and labels")
        self.input_ids = []
        self.token_type_ids = []
        self.mc_token_ids = []
        self.lm_labels = []
        self.mc_labels = []
        self.start_ids = []
        self.pad_token = tokenizer.convert_tokens_to_ids("<pad>")

        for dialog in tqdm(squad_data):
            context = dialog["context"].copy()
            if isinstance(context[0], list):
                random.shuffle(context)

            for utterance in dialog["utterances"]:
                history = utterance["history"]
                curr_input_ids = []
                curr_token_type_ids = []
                curr_mc_token_ids = []
                curr_lm_labels = []

                curr_utterances = utterance["candidates"]
                correct_idx = len(curr_utterances) - 1
                order = list(range(len(curr_utterances)))
                random.shuffle(order)
                correct_label = order.index(correct_idx)
                skip = False

                for j in range(len(curr_utterances)):
                    jdx = order[j]
                    lm_lab = jdx == correct_idx
                    candidate = curr_utterances[jdx]
                    instance = build_input_from_segments(
                        context, history, candidate, tokenizer, lm_labels=lm_lab
                    )

                    # skip inputs that are too long
                    if len(instance["input_ids"]) > args.max_len:
                        skip = True
                        break

                    curr_input_ids.append(instance["input_ids"])
                    curr_token_type_ids.append(instance["token_type_ids"])
                    curr_mc_token_ids.append(instance["mc_token_ids"])
                    curr_lm_labels.append(instance["lm_labels"])

                if skip:
                    continue

                self.start_ids.append(instance["start_idx"])
                self.mc_labels.append(correct_label)
                self.input_ids.append(curr_input_ids)
                self.token_type_ids.append(curr_token_type_ids)
                self.mc_token_ids.append(curr_mc_token_ids)
                self.lm_labels.append(curr_lm_labels)

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
