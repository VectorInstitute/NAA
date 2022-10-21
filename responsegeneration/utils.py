import json
import os
import re
import string
import warnings
from glob import glob
from itertools import chain
from operator import itemgetter
import torch
import random
from tqdm.notebook import tqdm
import torch.nn.functional as F

##################################
# Data Formatting Util Functions #
##################################

# special tokens used for the chatbot
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<eou>"],
}
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def add_special_tokens(model, tokenizer):
    """
    Add special tokens to the tokenizer and the model if they have not already been added.
    """
    orig_num_tokens = len(tokenizer.encoder)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_history(history_tokens, tokenizer):
    speakers = tokenizer.convert_tokens_to_ids(["<speaker1>", "<speaker2>"])

    # build speaker tokens
    speaker_tokens = []
    i = 0
    full_history = []
    for utterance in history_tokens:
        if utterance[0] not in tokenizer.convert_tokens_to_ids(
            ["<speaker1>", "<speaker2>"]
        ):
            speaker = speakers[i % 2]
            utterance = [speaker] + utterance
        else:
            speaker = utterance[0]
        speaker_tokens.extend([speaker] * len(utterance))
        full_history.extend(utterance)
        i = i + 1

    return full_history, speaker_tokens


def build_input_from_segments(
    context, history, reply, tokenizer, lm_labels=False, with_eos=True
):
    """
    Build a sequence of input from 3 segments: context, history and last reply.
    Inputs context, history, reply are in their original form.
    """
    # in case context hasn't been tokenized
    if isinstance(context[0], str):
        context = [tokenizer.encode(x) for x in context]
    elif isinstance(context, str):
        context = tokenizer.encode(context)

    if not isinstance(context[0], list):
        context = [context]

    full_history, speaker_tokens = build_history(history, tokenizer)
    if reply[0] not in tokenizer.convert_tokens_to_ids(["<speaker1>", "<speaker2>"]):
        next_speaker = tokenizer.convert_tokens_to_ids("<speaker2>")
        reply = [next_speaker] + reply
    else:
        next_speaker = reply[0]
    full_speakers_list = (
        [next_speaker] * (len(list(chain(*context))) + 1)
        + speaker_tokens
        + [next_speaker] * len(reply)
    )
    if with_eos:
        full_speakers_list = full_speakers_list + [next_speaker]

    bos, eos = tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
    first_part = [bos] + list(chain(*context)) + full_history
    second_part = reply + ([eos] if with_eos else [])
    sequence = first_part + second_part

    instance = {}
    instance["input_ids"] = sequence
    instance["token_type_ids"] = full_speakers_list
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * len(first_part)) + [-100] + second_part[1:]
    instance["start_idx"] = len(first_part)
    return instance


###########################
# Training Util Functions #
###########################


def average_distributed_scalar(scalar, args):
    """
    Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation.
    """
    if args.local_rank == -1:
        return scalar
    scalar_t = scalar / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_most_recent_checkpoint(directory):
    existing_checkpoints = glob(os.path.join(directory, "checkpoint*.pth"))
    if len(existing_checkpoints) == 0:
        return None
    p = re.compile("checkpoint_epoch(.*)_step")
    max_epoch = max([int(p.search(x)[1]) for x in existing_checkpoints])
    p = re.compile("checkpoint_epoch" + str(max_epoch) + "_step(.*).pth")
    max_step = max([int(p.search(x)[1]) for x in existing_checkpoints if p.search(x)])
    checkpoint = "checkpoint_epoch" + str(max_epoch) + "_step" + str(max_step) + ".pth"
    return os.path.join(directory, checkpoint)


def delete_prev_checkpoints(new_checkpoint, epoch):
    existing_checkpoints = glob(
        os.path.join(
            os.path.dirname(new_checkpoint), "checkpoint_epoch" + str(epoch) + "_step*"
        )
    )
    for c in existing_checkpoints:
        if c != new_checkpoint:
            os.remove(c)


############################
# Inference Util Functions #
############################


def top_filtering(
    logits, top_k=0.0, top_p=0.9, threshold=-float("Inf"), filter_value=-float("Inf")
):
    """Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
        top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
            whose total probability mass is greater than or equal to the threshold top_p.
            In practice, we select the highest probability tokens whose cumulative probability mass exceeds
            the threshold top_p.
        threshold: a minimal threshold to keep logits
    """
    assert (
        logits.dim() == 1
    )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(context, history, tokenizer, model, args, current_output=None):
    """
    Sample a sequence from the output of the model.

    Note that huggingface models do have a method for this already, but this version in particular
    has the args.force_answer option to toss out "I don't know" as the reply and take the next reply
    with the highest probability.
    """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = tokenizer.encode("<speaker2>")

    for i in range(args.max_length):
        instance = build_input_from_segments(
            context, history, current_output, tokenizer, with_eos=False
        )

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=args.device
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        logits = logits.logits

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits.clone(), top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        # can screw around with these generated tokens here
        if args.force_answer:
            prev = (
                torch.topk(probs, 3)[1]
                if args.no_sample
                else torch.multinomial(probs, 3)
            )
            if args.no_sample and (prev[0] == 40 or prev[0] == 314):
                print("Discarded 'I don't know' answer")
                idx = 1
                if prev[1] == 40 or prev[1] == 314:
                    idx = 2
            else:
                idx = 0
            prev = prev[idx]
        else:
            prev = (
                torch.topk(probs, 1)[1]
                if args.no_sample
                else torch.multinomial(probs, 1)
            )

        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn(
                        "Warning: model generating special token with probability 1."
                    )
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        current_output.append(prev.item())

    return current_output


def sample_sequence_tokens(input_ids, token_type_ids, tokenizer, model, args):
    """
    Sample a sequence from the output of the model, with input_ids as input

    Note that huggingface models do have a method for this already, but this version in particular
    has the args.force_answer option to toss out "I don't know" as the reply and take the next reply
    with the highest probability.
    """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = tokenizer.encode("<speaker2>")
    input_ids.extend(current_output)
    token_type_ids.extend(current_output)

    for i in range(args.max_length):

        input_ids_ts = (
            torch.tensor(input_ids, device=args.device)
            .type(torch.cuda.LongTensor)
            .unsqueeze(0)
        )
        token_type_ids_ts = (
            torch.tensor(token_type_ids, device=args.device)
            .type(torch.cuda.LongTensor)
            .unsqueeze(0)
        )

        if input_ids_ts.shape[-1] >= 1024:
            break

        logits = model(input_ids_ts, token_type_ids=token_type_ids_ts)
        logits = logits.logits

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits.clone(), top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        # can screw around with these generated tokens here
        if args.force_answer:
            prev = (
                torch.topk(probs, 3)[1]
                if args.no_sample
                else torch.multinomial(probs, 3)
            )
            if args.no_sample and (prev[0] == 40 or prev[0] == 314):
                print("Discarded 'I don't know' answer")
                idx = 1
                if prev[1] == 40 or prev[1] == 314:
                    idx = 2
            else:
                idx = 0
            prev = prev[idx]
        else:
            prev = (
                torch.topk(probs, 1)[1]
                if args.no_sample
                else torch.multinomial(probs, 1)
            )

        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn(
                        "Warning: model generating special token with probability 1."
                    )
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        current_output.append(prev.item())
        input_ids.append(prev.item())
        token_type_ids.append(prev.item())

    return current_output


#############################
# Evaluation Util Functions #
#############################


def normalize_text(s):
    """
    Removing articles and standardizing whitespace are all typical text processing steps.
    Punctuation is kept because if model predicts early period, it will probably predict <eos> next
    resulting in early termination, whereas commas don't seem to do that.
    """

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(s)))


def compute_f1(prediction, truth):
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(truth).split())

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def get_prediction(lm_logits, start_index, end_index):
    answer = torch.argmax(lm_logits[start_index:end_index], dim=1)
    return answer


def get_f1_score(input_ids, lm_logits, start_index, end_index, tokenizer):
    start_index = int(start_index.item())
    end_index = int(end_index.item())
    prediction = get_prediction(lm_logits, start_index, end_index)
    # readjust to exclude first speaker token and include eos
    answer = input_ids[start_index + 1 : end_index + 1]
    prediction = tokenizer.decode(prediction)
    answer = tokenizer.decode(answer)

    return compute_f1(prediction, answer)


####################################
#MSMARCO-Format Dataset Preparation#
####################################

def get_distractor(target_key, source_data):
    # get a distractor that is not the given answer and also not "No Answer Present."
    data = source_data
    all_keys = list(data['answers'].keys())
    # removing the target key from list of distractors
    all_keys.remove(target_key)
    idx = random.choice(all_keys)
    if data['answers'][idx][0] == 'No Answer Present.':
        return get_distractor(target_key, source_data)
    else:
        return data['answers'][idx][0]
    
def create_marco_format_QAdataset(data, categorical = True):
    all_keys = list(data['answers'].keys())
    new_data = []
    for key in tqdm(data['answers']):
        data_chunk = {}
        q_id = data['query_id'][key]
        question = data['query'][key]
        answers = data['answers'][key]
        better_answers = data['wellFormedAnswers'][key]
        contexts = data['passages'][key]
        question_type = data['query_type'][key]
        if categorical:
            category = data['category'][key]

        # if there exists a well-formed answer, use that one. Otherwise just use the answers provided.
        if isinstance(better_answers, list) and len(better_answers)!=0:
            new_answer = better_answers[0]
        else:
            new_answer = answers[0]

        # if there's multiple answers and a person has not provided a correct good response, skip this question
        if len(answers) > 1 and not isinstance(better_answers, list):
            continue

        # if the model can't extract the answer, at least be nice.
        if new_answer == 'No Answer Present.':
            new_answer = "I'm sorry, I don't know."

        data_chunk['query_id'] = q_id
        data_chunk['context'] = contexts
        data_chunk['query_type'] = question_type
        
        # if the dataset has category infromation
        if categorical:
            data_chunk['category'] = category
        
        
        data_chunk['utterances'] = []

        if new_answer == "I'm sorry, I don't know.":
            candidates = [get_distractor(key, data), get_distractor(key, data), new_answer]
        else:
            candidates = ["I'm sorry, I don't know.", get_distractor(key, data), new_answer]
        data_chunk['utterances'].append({
            'history': [question],
            'candidates': candidates
        })

        new_data.append(data_chunk)
    return new_data
