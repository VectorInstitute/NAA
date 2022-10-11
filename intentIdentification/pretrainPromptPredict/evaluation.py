from transformers import BertTokenizer, BertForMaskedLM
import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_json(data_file):
    if os.path.isfile(data_file):
        with open(data_file, "r") as read_file:
            data = json.load(read_file)
            return data


def load_list_file(list_file):
    with open(list_file, "r") as read_file:
        dialog_id_list = read_file.readlines()
        dialog_id_list = [l.strip("\n") for l in dialog_id_list]
        return dialog_id_list


dialog_data_file = "./outputMul2.2.json"
dialog_data = load_json(dialog_data_file)
dialog_id_list = list(set(dialog_data.keys()))

valid_list_file = "./MultiWOZ_2.1/valListFile.txt"
test_list_file = "./MultiWOZ_2.1/testListFile.txt"

valid_id_list = list(set(load_list_file(valid_list_file)))
test_id_list = load_list_file(test_list_file)
train_id_list = [
    did for did in dialog_id_list if did not in (valid_id_list + test_id_list)
]

# print('# of train dialogs:', len(train_id_list))
# print('# of valid dialogs:', len(valid_id_list))
# print('# of test dialogs :', len(test_id_list))
assert len(dialog_id_list) == len(train_id_list) + len(valid_id_list) + len(
    test_id_list
)

train_data = [v for k, v in dialog_data.items() if k in train_id_list]
valid_data = [v for k, v in dialog_data.items() if k in valid_id_list]
test_data = [v for k, v in dialog_data.items() if k in test_id_list]
assert len(train_data) == len(train_id_list)
assert len(valid_data) == len(valid_id_list)
assert len(test_data) == len(test_id_list)


# data = train_data + valid_data + test_data
data = test_data
# print(len(data))


ontology_file = "./MultiWOZ_2.1/ontology.json"
ontology_data = load_json(ontology_file)

# for ele in list(ontology_data.keys()):
#   print(ele)


questions = []
answers = []
overall_intents = []
domains_v1 = []
general_intents = []


def get_dst_diff(prev_d, crnt_d):
    diff = {}
    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        if v1 != v2:  # updated
            diff[k2] = v2
    return diff


def analyze_dialog(d, print_dialog=True):
    domains = []
    ignore_keys_in_goal = [
        "eod",
        "messageLen",
        "message",
    ]  # eod (probably) means the user archieved the goal.
    for dom_k, dom_v in d["goal"].items():
        if (
            dom_v and dom_k not in ignore_keys_in_goal
        ):  # check whether contains some goal entities
            domains.append(dom_k)
    #     print('{} domain(s): {}'.format(len(domains), domains))

    prev_user = ""
    if print_dialog:
        prev_d = None
        for i, t in enumerate(d["log"]):
            spk = (
                "Usr" if i % 2 == 0 else "Sys"
            )  # Turn 0 is always a user's turn in this corpus.
            if spk == "Sys":
                if prev_d is None:
                    prev_d = t["metadata"]
                else:
                    crnt_d = t["metadata"]
                    dst_diff = get_dst_diff(prev_d, crnt_d)

                    if t["dialog_act"] != {}:
                        get_intent = t["dialog_act"]
                        get_intent = list(get_intent.keys())[0]

                    curr_intents = ""
                    for domain, rest in dst_diff.items():
                        for intent in rest:
                            for key, value in rest[intent].items():
                                if (
                                    value != "not mentioned"
                                    and value != ""
                                    and value != []
                                ):
                                    #                             print(value)
                                    #                             print('Updated DST:', dst_diff)
                                    # print("CURRENT INTENTS", intent + " " + key)
                                    curr_intents = (
                                        curr_intents
                                        + domain
                                        + "-"
                                        + intent
                                        + "-"
                                        + key
                                        + " "
                                    )
                    if dst_diff != {} and curr_intents != "" and t["dialog_act"] != {}:
                        #                         if 'hospital' not in list(dst_diff.keys()) and 'bus' not in list(dst_diff.keys()):
                        #                         print("**************************")
                        #                         print('Updated DST:', t)
                        questions.append(u)
                        answers.append(t["text"])
                        domains_v1.append(list(dst_diff.keys()))
                        general_intents.append(get_intent)
                        overall_intents.append(curr_intents)
                    #
                    prev_d = crnt_d
            u = t["text"]
            # print('{}: {}'.format(spk, u))


for d in data:
    #     print('-' * 50)
    analyze_dialog(d, True)


generic_intents = []

for i in range(len(overall_intents)):
    sep_intent = overall_intents[i].split()
    for ele in sep_intent:
        x = ele.replace("-semi", "")
        x = x.replace("book-", "")
        if general_intents[i].split("-")[1] + "-" + x not in generic_intents:
            #             print(general_intents[i] + "-" + x)
            generic_intents.append(general_intents[i].split("-")[1] + "-" + x)

# print(generic_intents)
# print(len(generic_intents))


text = []

for i in range(len(questions)):
    #     curr = generic_intents[i].split("-")
    #     intent = generic_intents[0]
    #     domain = generic_intents[1]
    #     slot = generic_intents[2]
    curr = general_intents[i].split("-")
    text.append(
        questions[i]
        + " the intent is "
        + curr[1]
        + " the domain is "
        + domains_v1[i][0]
    )
# print(text[0:2])


class WOZDataset(Dataset):
    def __init__(self, text, tokenizer: BertTokenizer, max_token_len: int = 512):
        self.tokenizer = tokenizer
        self.data = text
        self.max_token_len = max_token_len

    def __getitem__(self, idx):
        data_row = self.data[idx]
        inputs = tokenizer.encode_plus(
            data_row,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs["labels"] = inputs.input_ids.detach().clone()
        for i in range(inputs.input_ids.shape[0]):
            curr_line = inputs.input_ids[i]
            for index in reversed(range(len(curr_line))):
                if curr_line[index] != 0:
                    curr_line[index - 1] = 103
                    curr_line[index - 5] = 103
                    curr_line[index - 6] = 103
                    break
        return dict(
            comment_text=data_row,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.labels,
        )

    def __len__(self):
        return len(self.data)


testing = WOZDataset(text, tokenizer)
# print(type(dataset))
# print(len(dataset))


# In[ ]:


# sample_item = testing[0]
# sample_item.keys()


# # In[ ]:


# sample_item["input_ids"].shape


# # In[ ]:


# loader = DataLoader(dataset, batch_size=16, shuffle=True)


# # In[ ]:


# sample_batch = next(iter(DataLoader(testing, batch_size=8, num_workers=2)))
# sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape


# # In[ ]:


# output = model(sample_batch["input_ids"], sample_batch["attention_mask"])


# In[ ]:


class WOZDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = WOZDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.test_dataset = WOZDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)


import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    BertForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


class LMModel(pl.LightningModule):
    def __init__(self, model_name_or_path, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.save_hyperparameters()
        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path, config=config)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    #         self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        loss = 0
        #         if labels is not None:
        #             loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, attention_mask, labels).loss
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, attention_mask, labels).loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, attention_mask, labels).loss
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


# In[ ]:


# trainer.fit(lmmodel, data_module)


# ## Evaluation

# In[ ]:


# import os
# import json
# import numpy as np

# def load_json(data_file):
#     if os.path.isfile(data_file):
#         with open(data_file, 'r') as read_file:
#             data = json.load(read_file)
#             return data

# def load_list_file(list_file):
#     with open(list_file, 'r') as read_file:
#         dialog_id_list = read_file.readlines()
#         dialog_id_list = [l.strip('\n') for l in dialog_id_list]
#         return dialog_id_list


# dialog_data_file = './outputMul2.2.json'
# dialog_data = load_json(dialog_data_file)
# dialog_id_list = list(set(dialog_data.keys()))

# valid_list_file = './MultiWOZ_2.1/valListFile.txt'
# test_list_file = './MultiWOZ_2.1/testListFile.txt'

# valid_id_list = list(set(load_list_file(valid_list_file)))
# test_id_list = load_list_file(test_list_file)
# train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]

# assert(len(dialog_id_list) == len(train_id_list) + len(valid_id_list) + len(test_id_list))

# train_data = [v for k, v in dialog_data.items() if k in train_id_list]
# valid_data = [v for k, v in dialog_data.items() if k in valid_id_list]
# test_data = [v for k, v in dialog_data.items() if k in test_id_list]
# assert(len(train_data) == len(train_id_list))
# assert(len(valid_data) == len(valid_id_list))
# assert(len(test_data) == len(test_id_list))


# data = train_data + valid_data + test_data


# questions = []
# answers = []
# overall_intents = []
# domains_v1 = []
# general_intents = []

# def get_dst_diff(prev_d, crnt_d):
#     diff = {}
#     for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
#         if v1 != v2: # updated
#             diff[k2] = v2
#     return diff

# def analyze_dialog(d, print_dialog=True):
#     domains = []
#     ignore_keys_in_goal = ['eod', 'messageLen', 'message'] # eod (probably) means the user archieved the goal.
#     for dom_k, dom_v  in d['goal'].items():
#         if dom_v and dom_k not in ignore_keys_in_goal: # check whether contains some goal entities
#             domains.append(dom_k)
# #     print('{} domain(s): {}'.format(len(domains), domains))

#     prev_user = ''
#     if print_dialog:
#         prev_d = None
#         for i, t in enumerate(d['log']):
#             spk = 'Usr' if i % 2 == 0 else 'Sys' # Turn 0 is always a user's turn in this corpus.
#             if spk == 'Sys':
#                 if prev_d is None:
#                     prev_d = t['metadata']
#                 else:
#                     crnt_d = t['metadata']
#                     dst_diff = get_dst_diff(prev_d, crnt_d)

#                     if t['dialog_act'] != {}:
#                         get_intent = t['dialog_act']
#                         get_intent = list(get_intent.keys())[0]

#                     curr_intents = ''
#                     for domain, rest in dst_diff.items():
#                       for intent in rest:
#                         for key, value in rest[intent].items():
#                           if value != 'not mentioned' and value != '' and value != []:
# #                             print(value)
# #                             print('Updated DST:', dst_diff)
#                             # print("CURRENT INTENTS", intent + " " + key)
#                             curr_intents = curr_intents + domain + "-" + intent + "-" + key + " "
#                     if dst_diff != {} and curr_intents != '' and t['dialog_act'] != {}:
# #                         if 'hospital' not in list(dst_diff.keys()) and 'bus' not in list(dst_diff.keys()):
# #                         print("**************************")
# #                         print('Updated DST:', t)
#                         questions.append(u)
#                         answers.append(t['text'])
#                         domains_v1.append(list(dst_diff.keys()))
#                         general_intents.append(get_intent)
#                         overall_intents.append(curr_intents)
# #
#                     prev_d = crnt_d
#             u = t['text']
#             # print('{}: {}'.format(spk, u))

# for d in data:
# #     print('-' * 50)
#     analyze_dialog(d, True)


# Loading Checkpoint

chk_path = "/h/elau/Conv_BERT/BERT_MLM/checkpoint_1/best-checkpoint.ckpt"

trained_model = LMModel.load_from_checkpoint(
    chk_path,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")
model = trained_model

# input = tokenizer.encode_plus(
#   sequence,
#   add_special_tokens=True,
#   max_length=512,
#   return_token_type_ids=False,
#   padding="max_length",
#   return_attention_mask=True,
#   return_tensors='pt',
# )

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trained_model = trained_model.to(device)

correct_intent = 0
correct_domain = 0
total = 0
for sentence in testing:
    input = sentence
    #     print(input['input_ids'])
    #     print(tokenizer.mask_token_id)
    #     print("HERE")
    #     print(torch.where(input['input_ids'] == tokenizer.mask_token_id)[0])
    #     quit()
    mask_token_index = torch.where(
        input["input_ids"].flatten() == tokenizer.mask_token_id
    )[0]

    #     print(tokenizer.decode(input['input_ids']))
    curr_labels = input["labels"]
    curr_text = input["comment_text"]
    #     logging.info(curr_text)
    #     print(curr_labels[0])
    #     print(curr_text)
    del input["labels"]
    del input["comment_text"]
    token_logits = model(input["input_ids"], input["attention_mask"])[1]
    print(mask_token_index)
    for ele in mask_token_index:
        #         print("CURRENT INDEX")
        #         print(ele)
        mask_token_logits = token_logits.logits[0, [ele], :]
        top_5_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()

        print(tokenizer.decode([curr_labels[0][mask_token_index[1]]]))

        for token in top_5_tokens:
            #             print(curr_labels[0][ele])
            print(tokenizer.decode([token]))
            if token == curr_labels[0][mask_token_index[1]]:
                logging.info(tokenizer.decode([token]))
                correct_intent += 1
            if token == curr_labels[0][mask_token_index[2]]:
                logging.info(tokenizer.decode([token]))
                correct_domain += 1
            if ele != mask_token_index[0]:
                total += 1

        print(curr_labels[0][mask_token_index[1]])

    logging.info("ACCURACY FOR CORRECT INTENT")
    logging.info(correct_intent / total)
    logging.info("ACCURACY FOR CORRECT INTENT")
    logging.info(correct_domain / total)
    logging.info("TOTAL")
    logging.info(total)

# chk_path = "/h/elau/Conv_BERT/BERT_MLM/checkpoints/best-checkpoint.ckpt"

# trained_model = LMModel.load_from_checkpoint(
#   chk_path,
# )

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")
# model = trained_model

# # sequence = "What are some recommendations for Indian restaurants? the intent is [MASK] the domain is [MASK]"
# # sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

# sequence = "Hi. How much is it for booking a hotel? the intent is [MASK] the domain is [MASK]"

# input = tokenizer.encode_plus(
#   sequence,
#   add_special_tokens=True,
#   max_length=512,
#   return_token_type_ids=False,
#   padding="max_length",
#   return_attention_mask=True,
#   return_tensors='pt',
# )

# mask_token_index = torch.where(input['input_ids'] == tokenizer.mask_token_id)[1]


# token_logits = model(input['input_ids'], input['attention_mask'])[1]
# for ele in mask_token_index:
#     mask_token_logits = token_logits.logits[0, [ele], :]
#     top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#     for token in top_5_tokens:
#         print(tokenizer.decode([token]))
#     print('='*80)
