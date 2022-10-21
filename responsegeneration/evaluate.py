import argparse
import json
import logging
import os
from pprint import pformat
from tqdm import tqdm

import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from haystack.nodes.evaluator.evaluator import semantic_answer_similarity

from DataLoader.dataloader import *
from Evaluation.bleu import Bleu
from Evaluation.rouge import Rouge
from utils import *

"""
NOTE: Bleu, Rouge, and SAS all work best with a SET of reference ground truth answers,
but the tokenized dataset file I made of MS Marco only includes one of answers from the original
dataset. It would be best to make a new kind of dataloader for this specific case, where there
are multiple ground truths for comparison.
"""

dontknow = "I'm sorry, I don't know."


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune GPT2")
    parser.add_argument(
        "--output_name",
        type=str,
        default="eval_results.json",
        help="Json file to save results to",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.getcwd(),
        help="Directory where training configuration, checkpoints, and arguments are saved",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Model checkpoint to run evaluation on",
    )
    parser.add_argument(
        "--valid_dataset_path", type=str, default="", help="validation dataset path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="marco",
        help="specify which Dataset file to use, depending on how your data is structured (marco, squad, ubuntu, general)",
    )

    cmd_line = parser.parse_args()
    return cmd_line


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.max_len = 900
    args.valid_batch_size = 1
    args.second_loss = "mc"
    train_args = torch.load(os.path.join(args.log_dir, "model_training_args.bin"))
    inference_args = argparse.Namespace(
        no_sample=True,
        max_length=200,
        min_length=1,
        seed=39,
        temperature=0.7,
        top_k=100,
        top_p=0.0,
        device=args.device,
        force_answer=False,
    )

    # setting up logging
    logger = logging.getLogger("FinetuningGPT2")
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare tokenizer and model.")
    # datasets were initialized originally on the gpt2 tokenizer so always use that one
    tokenizer = GPT2Tokenizer.from_pretrained(args.log_dir)

    # Initializing pretrained model
    config = GPT2Config.from_json_file(os.path.join(args.log_dir, "config.json"))
    state_dict = torch.load(args.checkpoint)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model = GPT2LMHeadModel.from_pretrained(
        config._name_or_path, config=config, state_dict=state_dict
    )
    model.to(args.device)
    model.eval()
    del state_dict

    # add our special tokens to the model
    add_special_tokens(model, tokenizer)

    # init dataloader for validation set
    valid_loader = get_validation_dataloader(args, tokenizer)

    logger.info("Beginning Evaluation")
    metrics = {"EM": [], "F1": []}
    gts = []
    predictions = []
    answeredwhenshouldnt = 0
    didntwhenshould = 0
    sas_model = "cross-encoder/stsb-roberta-large"
    for step, data in tqdm(enumerate(valid_loader)):
        # unpack inputs
        b, nc, ln = data["input_ids"].shape
        if int(data["mc_labels"][0]) >= nc:
            continue
        if int(data["reply_start"][0]) >= ln:
            continue

        gt_tokens = list(
            data["input_ids"][
                0, int(data["mc_labels"][0]), int(data["reply_start"][0]) :
            ]
        )
        input_ids = list(
            data["input_ids"][
                0, int(data["mc_labels"][0]), : int(data["reply_start"][0])
            ]
        )
        token_type_ids = list(
            data["token_type_ids"][
                0, int(data["mc_labels"][0]), : int(data["reply_start"][0])
            ]
        )

        with torch.no_grad():
            output = sample_sequence_tokens(
                input_ids, token_type_ids, tokenizer, model, inference_args
            )

        # decode then encode to get rid of special tokens
        gt_reply = tokenizer.decode(gt_tokens, skip_special_tokens=True)
        pred = tokenizer.decode(output, skip_special_tokens=True)
        gts.append(gt_reply)
        predictions.append(pred)

        if gt_reply == dontknow and pred != dontknow:
            answeredwhenshouldnt += 1
        if pred == dontknow and gt_reply != dontknow:
            didntwhenshould += 1

        # calculate Exact-Match and F1-Scores here
        em = int(gt_reply == pred)
        f1 = compute_f1(pred, gt_reply)

        metrics["EM"].append(em)
        metrics["F1"].append(f1)

    metrics["Answered When Shouldnt"] = answeredwhenshouldnt
    metrics["Did Not Answer When Should"] = didntwhenshould

    print("Calculating additional metrics")

    # the format the MS Marco evaluation scripts want
    refs = {i: [x] for i, x in enumerate(gts)}
    cans = {i: [x] for i, x in enumerate(predictions)}

    bleu_scores, _ = Bleu(4).compute_score(refs, cans)
    rouge_score, _ = Rouge().compute_score(refs, cans)
    for i, bleu_score in enumerate(bleu_scores):
        metrics["Bleu-%d" % (i + 1)] = bleu_score
    metrics["Rouge-L"] = rouge_score

    # get SAS
    # currently only one pred and gt per row, so top1 and topk are identical
    top1sas, topksas = semantic_answer_similarity(
        predictions=[[pred] for pred in predictions],
        gold_labels=[[gt] for gt in gts],
        sas_model_name_or_path=sas_model,
    )
    metrics["SAS"] = top1sas

    for key in metrics:
        if isinstance(metrics[key], list):
            metrics[key] = sum(metrics[key]) / len(metrics[key])

    logger.info("Writing to file")
    write_data = {"metadata": vars(args), "metrics": metrics}
    with open(args.output_name, "w") as f:
        json.dump(write_data, f, indent=4)
