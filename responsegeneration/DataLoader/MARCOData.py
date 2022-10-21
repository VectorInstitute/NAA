from itertools import chain
import ijson
import random
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import *
from DataLoader.dataset_utils import *

# 'I'm sorry, I don't know' encoded by the gpt2 tokenizer
dontknow = [40, 1101, 7926, 11, 314, 836, 470, 760, 13]


class MARCODataset(Dataset):
    def __init__(self, args, tokenizer, dataset_path, limit=None):
        """
        Initialize dataset for the Microsoft MARCO dataset.
        """
        if args.second_loss == "bc":
            raise ValueError(
                "Using binary classification as the second head loss is not supported by the MARCO Dataset class currently"
            )

        print("Loading Dataset")
        # using ijson because MARCO is huge
        data_stream = ijson.items(open(dataset_path, "r"), "item")
        marco_data = (o for o in data_stream)

        print("Building inputs and labels")
        self.input_ids = []
        self.token_type_ids = []
        self.mc_token_ids = []
        self.lm_labels = []
        self.mc_labels = []
        self.start_ids = []
        self.pad_token = tokenizer.convert_tokens_to_ids("<pad>")

        # MS Marco has around 33% of answers as "I don't know", and the model picks up a bad habit from it
        # Limiting the amount of those in the dataset helps it provide accurate answers when possible more often
        num_dunno = 5000

        # used to limit size, I did this for the validation set just so it would be more reasonable
        # training time
        so_far = 0
        dunno_so_far = 0
        for dialog in tqdm(marco_data):
            # create context
            extra_contexts = 5
            necessary = [
                x["passage_text"] for x in dialog["context"] if x["is_selected"] == 1
            ]
            extras = [
                x["passage_text"] for x in dialog["context"] if x["is_selected"] == 0
            ]
            full_context = necessary + extras[:extra_contexts]
            total_len = (
                len(list(chain(*full_context)))
                + len(dialog["utterances"][0]["history"])
                + max([len(x) for x in dialog["utterances"][0]["candidates"]])
            )
            while total_len > args.max_len and extra_contexts > 1:
                extra_contexts = extra_contexts - 1
                full_context = necessary + extras[:extra_contexts]
                total_len = (
                    len(list(chain(*full_context)))
                    + len(dialog["utterances"][0]["history"])
                    + max([len(x) for x in dialog["utterances"][0]["candidates"]])
                )
            if extra_contexts < 2:
                # not enough distractors in context, skip
                continue

            if total_len > args.max_len:
                full_context = necessary + extras[:extra_contexts]
            context = full_context

            if isinstance(context[0], list):
                random.shuffle(context)

            for utterance in dialog["utterances"]:
                history = utterance["history"]
                curr_input_ids = []
                curr_token_type_ids = []
                curr_mc_token_ids = []
                curr_lm_labels = []

                # get possible
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
                    if len(candidate) == 0:
                        # skip empty replies
                        skip = True
                        break
                    if candidate == dontknow and lm_lab:
                        # only keep a set number of "I don't know"s in the dataset
                        if dunno_so_far >= num_dunno:
                            skip = True
                            break
                        else:
                            dunno_so_far += 1

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

            # used to load only a portion of the data
            so_far = so_far + 1
            if limit is not None and so_far == limit:
                break

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
