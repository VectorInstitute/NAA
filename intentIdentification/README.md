
## Intent Identification

Identifying and understanding the intent of customers’ query and input plays an important role in the performance of goal-oriented NAA for customer service support. In this repo, we frame the backend of the Intent Identification module of NAA as a classfier that estimates customer’s intentions. The binary classifier identifies whether the question is general or request from the user. The multi-classifier is a query encoder that classifies queries into N classes. Please download supplementary materials including checkpoints and data from [here](https://drive.google.com/drive/folders/10gGCBdVBfKd3DUI_0XqR28hlThWm8Ek_?usp=sharing).

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation
1. Clone the repo
```sh
git clone xxx
```
2. Create and activate virtual environment 
```sh
python -m venv env
source env/bin/activate
```
3. install the required libraries
```sh
pip install -r requirements.txt 
```
### Data Format 
Both models are trained on intents classification dataset, Banking77, which is obtained from https://github.com/PolyAI-LDN/task-specific-datasets . With few-shot learning, we used "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction" dataset, specifically the domain credit_cards.

Data should be formatted in a csv. The following shows an example on the expected format. 

  
text  | category
------------- | -------------
How can I receive money?  | receiving_money
where can I change my address?	  | edit_personal_details
What do I do if I still have not received my new card?	  | card_arrival
Is there anywhere I can't use my card?		  | card_acceptance
I want to have multiple currencies in my account if possible.		  | fiat_currency_support


### Training the Model
This project composed of few fine tuning scripts. 

`multi_classification/text_classification_with_BERT.ipynb` is for multi-class classification. 

`multi_classification/fewshot_text_classification_with_BERT.ipynb` is for few-shot learning classification. 

`binary_classification/general_text_classification_with_BERT.ipynb` is a binary classification for identifying whether the question is general or request from the user. 


To convert Jupyter notebook to python script. We can do 
```
jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```

`train.sh` are provided as outlines for how to use the training scripts.

### Interactive and Visualization

The interactive script is `interactive.ipynb`.
For the binary classification the checkpoint can be downloaded from [here](https://drive.google.com/file/d/1Asg4vsnyUThZ3sjW9ricYcnZ1i3nXqTq/view?usp=sharing). 


<!-- CONTACT -->
## Contact

Elaine Lau - https://github.com/yunglau - tsoi.lau@mail.mcgill.ca
