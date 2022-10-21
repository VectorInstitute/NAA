# ConversationalAI: Question-Answering

Author: Shirley Wang, 2021

The Task-Guided Question-Answering 
involves inputting a context and a question, and having the model output the answer to the question by extracting it from the context. We have decided to combine these tasks due to how training purely on dialogue datasets does not provide the model knowledge needed to correctly answer questions, and being able to fully finetune a giant GPT3 for specific needs is costly and not very accessible. By taking advantage of how question-answering models simply extract answers from a context, we can use that to provide answers to customer queries in our model. 

The best dataset to train the model on would be a question-answering dataset where all answers are in a human dialogue full sentence format, instead of purely extractive answers. The one we found was the MS Marco dataset, which provides answers as full sentences, and most answers tried to not reuse language from the context.

The model used is GPT2 from huggingface. You can specify which size you would like to use, or if you would like to reload a DialoGPT checkpoint instead. Larger models have been shown to be able to learn more and perform better, so that's something to keep in mind. I would recommend trying to use at least GPT2-medium. GPT2-Large is difficult to train unless you have a 24GB GPU, and train with mixed precision and batch size 1.

## Environment Setup

We recommend Python 3.6 or higher, PyTorch 1.8.0 or higher and transformers v4.6.0 or higher. You can set up your virtual environment using pip and virtual-env by the commands:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Important: the training makes use of pytorch-ignite, but uses regular pytorch's distributed backend for distributed training. Pytorch-ignite eventually released their own distributed package, which turns out to be incompatible with using regular pytorch distributed functions. So it's important to use pytorch-ignite 0.4.0 or a version before that.

It's probably worth either rewriting the training script in pytorch-ignite or regular pytorch in the future for consistency?

## Data Format

### General Data Format

Data should be formatted in a json. The `GeneralData.py` class is provided, which will work for finetuning a model on any existing in this format:

```json
[
    {
    "context": [
        "information1",
        "information2",
        ...
    ],
    "history": [
        "Question"
    ],
    "reply": "answer to question"
  },
  ...
]
```

This data format works for training on both generic dialogue data, and QA data.

Context is a list of different chunks of information, where hopefully one will contain information to answer the question. There may only be one context paragraph, which if that's the case there's only one entry in the list. It's a list so that in case there is information coming from many different sources, it's easier to compile together. In the case of generic dialogue, you can either put nothing here, or put a persona the model should try to adopt when answering the questions.

History refers to the "conversation history". The last entry should be the question being asked to the model, or what the human said that the model should try to reply to. It's called history and in a list format in case you want to try training the model to be more dynamic based on previous inputs and replies, like if you want to train the model to be able to ask clarifying questions and make use of that information, this would be useful. Otherwise if you just want straightforwards QA, history should just be one entry which is the question.

Reply is the answer to the question, or the reply to the last thing the user inputted.

IMPORTANT: distractors are not implemented in this dataset class, so it will only train the language modelling part of the model. If you want distractors to make use of the second head on the model, you will probably want to create your own Dataset class.

### Specific Cases

If the general format doesn't work for your case, you can try to create your own Dataset class for whatever data you have and use that instead. If that's the case, put it in the `DataLoader` folder and add it to the case statement in `DataLoader/dataloader.py`. `UbuntuData.py`, `SQuADData.py`, `MARCOData.py` are all existing data format files for specific datasets we attempted using, and can serve as examples to what the data loading format should be.

`UbuntuData.py`: Corresponds to the Ubuntu Dialogue Corpus. A purely dialogue dataset. Also contains the ability format data in either multiple choice or binary classification for the second head.
    - BC version probably isn't compatible with resuming training from a checkpoint in the middle of the epoch.

`SQuADData.py`: Corresponds to the Stanford Question-Answering dataset. A purely QA dataset that also includes topic.

`MARCOData.py`: Corresponds to the Microsoft MARCO dataset. A purely QA dataset, that also contains well-formed answers.

I recommend tokenizing your data files and saving them to new files before doing training, instead of tokenizing inputs during training, just to speed things up somewhat. Currently tokenizing as training goes along is not supported in the code.

### Formatted Datasets

You can download the original datasets from their websites. We also provide the datasets reformatted so our dataloaders can work with them here. These are the untokenized version, so you will have to tokenize the contents yourself using huggingface's gpt2 tokenizer, or any other tokenizer. The TCard dataset is a small json file of QA scraped from the UofT Tcard website, as an example of how to further finetune a model on a small custom domain specific dataset.
- Ubuntu Dialogue Corpus: [dataset](https://drive.google.com/file/d/17JrsvseD_D0qOiouNubZ3w6tpCABvk3e/view?usp=sharing)
- SQuAD: [train](https://drive.google.com/file/d/1YvT55Lc7f_PpjOTHqIvM0r1abyLXEA1Z/view?usp=sharing) | [valid](https://drive.google.com/file/d/1vKNQ2Gh-YulPgIWsPoQvv_m0-krm9XnB/view?usp=sharing)
- MS Marco: [train](https://drive.google.com/file/d/1S6GMZxH66dbXCDNjzMLVHCY5T5QN4LDd/view?usp=sharing) | [valid](https://drive.google.com/file/d/1ApcxNrfRFVlGlrmllEWr4Ve776tTPKEQ/view?usp=sharing)
- TCard: [dataset](https://drive.google.com/file/d/15G9IMiS7ibyTR5NIEtBSHJEg4PvZqAaK/view?usp=sharing)
- Eli-5 [train]() | [valid]()

## Training the Model

The training script is `finetune_gpt.py`. 

```bash
python finetune_gpt.py 
    --log_dir [save_path] \
    --model_type [huggingface pretrained model name] \
    --load_checkpoint [path to existing checkpoint] \
    --second_loss [mc or bc] \
    --train_batch_size [integer] \
    --valid_batch_size [integer] \
    --train_dataset_path [path to train data json] \
    --valid_dataset_path [path to train data json] \
    --dataset_type [which dataset file] \
    --max_len [maximum length of input] \
    --num_gpus [number of gpus for use] \
    --lr [learning rate] \
    --lm_coef [language modelling loss coefficient] \
    --mc_coef [second head loss coefficient] \
    --max_norm [maximum gradient norm] \
    --gradient_accumulation_steps [integer] \
    --n_epochs [number of training epochs] \
    --random_seed [integer] \
    --eval_before_start \
    --fp16
```
Notes on inputs:
- `model_type` is what goes into the initialization of the model `GPT2DoubleHeadsModel.from_pretrained(args.model_type)`, you can input 'GPT2', 'GPT2-medium', 'GPT2-large', 'microsoft/DialoGPT-large', etc. View huggingface to see their available pretrained models.
- `load_checkpoint` is specifically for the case where you have an existing checkpoint on your system that you want to restart training from. This is useful for when you've already finetuned a model, and want to further finetune it on a different dataset.
- `second_loss` refers to if the second head of the DoubleHeads model will use multiple choice classification (mc) or binary classification loss (bc).
    - Multiple Choice Classification is where each input is grouped together with some distractors (bad replies), and has to choose from all of them which is the best reply.
    - Binary Classification instead treats each input as independant, and predicts whether the reply is correct or not.
    - Both involve distractors being mixed into the dataset.
- `dataset_type` controls which Dataset file will be used to load the data. Currently supported is 'marco', 'squad', 'ubuntu', or 'general', which are explained further in the Data Format section.
- `max_len` controls what the maximum length of an input will be. Any inputs longer than the max len specified will be thrown out. This is to help with controlling the CUDA memory to prevent out of memory errors.
- `num_gpus` is mainly used to identify if we are doing distributed training or not. Specifying a number larger than 1 will set it into distributed training mode. Note that if that is the case, the environment variable LOCAL_RANK should get set for each process (which is automatically set with torch.distributed.run)
- `lm_coef` and `mc_coef` are the weights for each head's loss to form the overall loss. 
- `gradient_accumulation_steps` specifies how many steps to go through before performing a backpropagation and step. This is useful for cases where we are forced to use a small batch size, so setting this to a number other than 1 helps simulate a larger batch size.
- `eval_before_start`: using this flag will run the validation step before training starts.
- `fp16`: using this flag will have training use torch's automatic mixed precision. Very useful for training large models. 

`train.sh` and `dist_train.sh` are provided as outlines for how to specify and use the training scripts.

## Interacting with the Model

The `Interact.ipynb` notebook is provided as a way to interact with a trained model. Further notes are inside the notebook. You can download one of our pretrained models from the section below.

Since the model just makes use of huggingface's GPT2 model, you can also use huggingface's built in `generate()` method to use any different language generation methods, like beam search or greedy search. 

If you just want to play around with the model, I recommend the GPT2-Medium one trained on MS Marco.

## Evaluation and Pretrained Models

Evaluation metrics are tricky, since we are trying to do QA while also maintaining a dialogue-esque format. Because of the dialogue format of the answer, traditional QA metrics aren't very good. 

For pure QA, Exact Match and F1-Score are the usual metrics. 
- Exact Match: An exact match between ground truth and predicition, 1 for they match and 0 for any difference. Not the best, since a slightly differently phrased prediction could still be correct but EM would give a bad score.
- F1-Score: The harmonic mean of precision and recall between ground truth and prediction. Technically a better metric. But it doesn't account for how

NOTE: F1-Score is tracked during training, but the training F1-Score is prone to giving a higher value than what should actually be there. For the training calculation of F1 Score, I'm just taking the argmax for each position in the logits, where as at inference time, the language generation model would generate the next token one at a time. Because of that, during response generation time, it's very common to predict an "eos" following a period, but that's not visible here when just taking the argmax, and what would have been predicted provided no eos is visible. So there's a higher chance of accurate tokens appearing here in the F1 Score than won't get predicted during actual response generation time. This is probably due to it trying to also capture how language works on top of extracting the right part from the context. Running the evaluation script will be more accurate than using the F1-Score given during training.

The MS Marco dataset also has a leaderboard that tracks Rouge-L and Bleu-L scores on it, although it was retired in 2020 and no longer maintained. For fairness, we also calculate those scores here. The evaluation script is also calculating the Semantic Answer Similarity (from https://arxiv.org/abs/2108.06130).

NOTE: The context in the Marco dataset is created from a random sample of the passages, since larger models can't handle a super large input size on 1 GPU. As a result, multiple runs of the evaluation script will result in slightly different numbers.

NOTE: Models trained on the Ubuntu dataset do not have evaluation metrics since they are Dialogue models instead of QA models.

NOTE: Models are evaluated on the validation set of the dataset they were trained on. That is why there is such a difference in the metrics of a model trained on SQuAD vs the metrics of a model trained on MS Marco.

The models trained on MS Marco actually perform very well in a zero-shot setting on completely new contexts, even though the evaluation metrics are low. Starting with the usual pretrained GPT2 seems to do better than starting with DialoGPT (which was already trained on dialogue). The models trained on SQuAD and Ubuntu are also present here, although they're not recommended. SQuAD is pure QA, so it does not capture that dialogue aspect at all, and Ubuntu is pure dialogue, so it's incapable of answering questions.

| Model           | Dataset          | Test Set | Epochs | EM    | F1    | Bleu-1 | Rouge-L | SAS   | Download Here |
|-----------------|------------------|----------|--------|-------|-------|--------|---------|-------|---------------|
| GPT2-Medium     | MS Marco         |   Eli5   | 5      |   -   | 6.95% |  0.08% |  5.24%  | 30.44%|       -       |
| GPT2-Medium     | MS Marco + Eli5  |   Eli5   | 5 + 10 |   -   | 12.96%|  9.44% |  9.29%  | 37.63%|       -       |
| GPT2-Medium     | MS Marco         | MS Marco | 5      | 7.6%  | 40.2% | 32.0%  | 36.0%   | 53.1% | [here](https://drive.google.com/file/d/13gz-4cekfrFX2n0Q6vjGjL0fs9AJOPPW/view?usp=sharing) |
| GPT2-Medium     | MS Marco + TCard | MS Marco | 5 + 20 | 7.8%  | 42.3% | 36.6%  | 37.8%   | 55.6% | [here](https://drive.google.com/file/d/1RXTYmfHjc157s8XyrHbQzA5p99cHjp4g/view?usp=sharing) |
| DialoGPT-Medium | MS Marco         | MS Marco | 5      | 4.4%  | 28.2% | 15.3%  | 24.6%   | 43.7% | [here](https://drive.google.com/file/d/19gjS0YEetC3lJjw79GFoOgwOFqUvYK2z/view?usp=sharing) |
| GPT2-Large      | MS Marco         | MS Marco | 5      | 5.8%  | 32.0% | 20.4%  | 28.0%   | 46.8% | [here](https://drive.google.com/file/d/1uMi43m8jmRRaAu6kiVsBnka09WLo-GoP/view?usp=sharing) |
| GPT2-Medium     | SQuAD            |  SQuAD   | 3      | 60.0% | 69.4% | 68.0%  | 68.7%   | 69.9% | [here](https://drive.google.com/file/d/1hSRTjOdztmIbb6peQ4Km_MIfbcMTPz6J/view?usp=sharing) |
| GPT2-Small (BC) | Ubuntu           |  Ubuntu  | 3      | - | - | - | - | - | [here](https://drive.google.com/file/d/1UCDaNynb4U2ZI1mPxyDhvp_uLrvRI51y/view?usp=sharing) |
| GPT2-Small (MC) | Ubuntu           |  Ubuntu  | 3      | - | - | - | - | - | [here](https://drive.google.com/file/d/10KJauuiwrWE_grSDx5I17P9OhVcDeBWq/view?usp=sharing) |

Further finetuning an MS Marco dataset model on a custom domain specific dataset does not impact its generalization capabilities to overall QA, while also improving its abilities to answer question in that domain. You can see this in the GPT2-Medium + TCard model, where after training it for 5 epochs on MS Marco, it was further finetuned on the TCard dataset for 20 epochs, and its evaluation numbers on the MS Marco validation dataset did not decrease (the TCard dataset is too small to have a test set).

You can run the evaluation script with:

```
python ./evaluate.py \
    --output_name [json file name] \
    --log_dir [training directory] \
    --checkpoint [checkpoint filename] \
    --valid_dataset_path [validation dataset json file] \
    --dataset_type [see "Training the Model" for what dataset_type refers to]
```

See `evaluate.sh` for an example. To use it, update the parameters of the script.

## Contact

Shirley Wang - https://github.com/ShirleyWangCVR - shirleywang@hotmail.ca
