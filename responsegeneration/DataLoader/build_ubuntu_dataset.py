"""
Using the original csvs created from the ubuntu-ranking-dataset-creator, 
creates json versions for usage.

TODO: needs to get tested
"""

import tqdm
import json
import pandas as pd


def process_context(text, tokenizer):
    """
    Processes the history from the ubuntu dataset.

    Returns: full history, speaker tokens list, next speaker
    """
    # the if x gets rid of empty strings
    history = [x for x in text.strip().split("__eot__") if x]
    speakers = ["<speaker1>", "<speaker2>"]

    history = [x for x in text.strip().split("__eot__") if x]
    speakers = ["<speaker1>", "<speaker2>"]

    full_history = [
        speakers[i % 2] + " " + history[i].strip() for i in range(len(history))
    ]
    orig_history = full_history
    full_history = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in full_history
    ]
    speaker_tokens = [speakers[i % 2] for i, s in enumerate(full_history) for _ in s]
    next_speaker = len(history) % 2
    return full_history, speaker_tokens, speakers[next_speaker], orig_history


def get_json_output(persona, history, reply, tokenizer, distractor):
    full_history, speaker_tokens, next_speaker, orig_history = process_context(
        history, tokenizer
    )
    full_reply = next_speaker + " " + reply.strip()
    full_distractor = next_speaker + " " + distractor.strip()

    new_output = {
        "personality": ["i work at Ubuntu .", "i am a professional ."],
        "utterances": [
            {"candidates": [full_distractor, full_reply], "history": orig_history}
        ],
    }

    return new_output


### Training Set ###

ubuntu_df = pd.read_csv("../../ubuntu-ranking-dataset-creator/src/train.csv")
ubuntu_df.head()

# use token <eou> instead of __eou__ for consistency
ubuntu_df["Context"] = ubuntu_df["Context"].str.replace("__eou__", "<eou>")
ubuntu_df["Utterance"] = ubuntu_df["Utterance"].str.replace("__eou__", "<eou>")

distractors = ubuntu_df.sample(frac=1).reset_index()
distractors = list(distractors["Utterance"])
histories = list(ubuntu_df[ubuntu_df["Label"] == 1]["Context"])
replies = list(ubuntu_df[ubuntu_df["Label"] == 1]["Utterance"])
print(len(histories))

# build training dataset
training_stuff = []
persona = ["i work at Ubuntu .", "i am a professional ."]
for i in tqdm(range(len(histories))):
    json_out = get_json_output(
        persona, histories[i], replies[i], tokenizer, distractors[i]
    )
    training_stuff.append(json_out)


### Validation and Test Set ###

# validation and test set are structured this way
ubuntu_df = pd.read_csv("../../ubuntu-ranking-dataset-creator/src/valid.csv")
ubuntu_df.head()

# use token <eou> instead of __eou__ for consistency
ubuntu_df["Context"] = ubuntu_df["Context"].str.replace("__eou__", "<eou>")
ubuntu_df["Ground Truth Utterance"] = ubuntu_df["Ground Truth Utterance"].str.replace(
    "__eou__", "<eou>"
)
ubuntu_df["Distractor_0"] = ubuntu_df["Distractor_0"].str.replace("__eou__", "<eou>")

# get dataset stuff
distractors = list(ubuntu_df["Distractor_0"])
histories = list(ubuntu_df["Context"])
replies = list(ubuntu_df["Ground Truth Utterance"])
print(len(histories))

# build validation dataset
valid_stuff = []
persona = ["i work at Ubuntu .", "i am a professional ."]
for i in tqdm(range(len(histories))):
    json_out = get_json_output(
        persona, histories[i], replies[i], tokenizer, distractors[i]
    )
    valid_stuff.append(json_out)

# build test dataset
ubuntu_df = pd.read_csv("../../ubuntu-ranking-dataset-creator/src/test.csv")

# use token <eou> instead of __eou__ for consistency
ubuntu_df["Context"] = ubuntu_df["Context"].str.replace("__eou__", "<eou>")
ubuntu_df["Ground Truth Utterance"] = ubuntu_df["Ground Truth Utterance"].str.replace(
    "__eou__", "<eou>"
)
ubuntu_df["Distractor_0"] = ubuntu_df["Distractor_0"].str.replace("__eou__", "<eou>")

# get dataset stuff
distractors = list(ubuntu_df["Distractor_0"])
histories = list(ubuntu_df["Context"])
replies = list(ubuntu_df["Ground Truth Utterance"])
print(len(histories))

test_stuff = []
persona = ["i work at Ubuntu .", "i am a professional ."]
for i in tqdm(range(len(histories))):
    json_out = get_json_output(
        persona, histories[i], replies[i], tokenizer, distractors[i]
    )
    test_stuff.append(json_out)

new_dataset = {"train": training_stuff, "valid": valid_stuff, "test": test_stuff}

with open("ubuntu_data.json", "w") as outfile:
    json.dump(new_dataset, outfile, indent=2)
