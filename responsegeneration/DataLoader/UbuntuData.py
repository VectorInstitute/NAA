import random
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import *
from DataLoader.dataset_utils import *


class UbuntuBCDataset(Dataset):
    """
    Ubuntu Dialogue Corpus dataset formatted for binary classification second head.
    """

    def __init__(self, args, tokenizer, dataset_path):
        """
        Initialize dataset for ubuntu data
        """
        ubuntu_data = get_tokenized_dataset(tokenizer, dataset_path)

        print("Building inputs and labels")
        self.input_ids = []
        self.token_type_ids = []
        self.mc_token_ids = []
        self.lm_labels = []
        self.mc_labels = []
        self.start_ids = []
        self.pad_token = tokenizer.convert_tokens_to_ids("<pad>")

        self.correct_replies = []
        self.incorrect_replies = []

        idx = 0
        for dialog in tqdm(ubuntu_data):
            persona = dialog["personality"].copy()
            random.shuffle(persona)

            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2 * args.max_history + 1) :]

                for j, candidate in enumerate(utterance["candidates"]):
                    # last candidate is correct one
                    lm_labels = j == (len(utterance["candidates"]) - 1)
                    instance = build_input_from_segments(
                        persona, history, candidate, tokenizer, lm_labels=lm_labels
                    )

                    if len(instance["input_ids"]) > args.max_len:
                        continue

                    self.input_ids.append(instance["input_ids"])
                    self.token_type_ids.append(instance["token_type_ids"])
                    self.mc_token_ids.append(instance["mc_token_ids"])
                    self.lm_labels.append(instance["lm_labels"])
                    self.mc_labels.append(int(lm_labels))  # 1 if correct, 0 if wrong
                    self.start_ids.append(instance["start_idx"])

                    if lm_labels:
                        self.correct_replies.append(idx)
                    else:
                        self.incorrect_replies.append(idx)
                    idx = idx + 1

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


class UbuntuMCDataset(Dataset):
    """
    Ubuntu Dialogue Corpus dataset.
    """

    def __init__(self, args, tokenizer, dataset_path):
        """
        Initialize dataset for ubuntu data
        """
        ubuntu_data = get_tokenized_dataset(tokenizer, dataset_path)

        print("Building inputs and labels")
        self.input_ids = []
        self.token_type_ids = []
        self.mc_token_ids = []
        self.lm_labels = []
        self.mc_labels = []
        self.pad_token = tokenizer.convert_tokens_to_ids("<pad>")

        self.correct_replies = []
        self.incorrect_replies = []

        for dialog in tqdm(ubuntu_data):
            persona = dialog["personality"].copy()
            random.shuffle(persona)

            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2 * args.max_history + 1) :]
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
                        persona, history, candidate, tokenizer, lm_labels=lm_lab
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
