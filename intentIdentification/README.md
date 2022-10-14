# Conversational AI

<!-- PROJECT LOGO -->

<!-- ABOUT THE PROJECT -->
## About The Project

Identifying and understanding the intent of customers’ query and input plays an important role in the performance of goal-oriented NAA for customer service support. In this repo, we frame the backend of the Intent Identification module of NAA as a classfier that estimates customer’s intentions. The binary classifier identifies whether the question is general or request from the user. The multi-classifier is a query encoder that classifies queries into N classes. 

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

`predefinedClassification/text_classification_with_BERT.ipynb` is for multi-class classification. 

`predefinedClassification/fewshot_text_classification_with_BERT.ipynb` is for few-shot learning classification. 

`binary_classification/general_text_classification_with_BERT.ipynb` is a binary classification for identifying whether the question is general or request from the user. 


To convert Jupyter notebook to python script. We can do 
```
jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```

`train.sh` are provided as outlines for how to use the training scripts.

### Interactive and Visualization

The interactive script is `interactive.ipynb`.

### Evaluation 

Model-Dataset  | F1-Score
------------- | -------------
Predefined Classes - Banking77  | 0.92
Few-Shot Learning - OOS	- 1 shot  | 0.85
Few-Shot Learning - OOS	- 5 shot  | 0.90

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- Citations --> 
## References 

```bibtex
@inproceedings{CoopeFarghly2020,
    Author      = {Sam Coope and Tyler Farghly and Daniela Gerz and Ivan Vulić and Matthew Henderson},
    Title       = {Span-ConveRT: Few-shot Span Extraction for Dialog with Pretrained Conversational Representations},
    Year        = {2020},
    url         = {https://arxiv.org/abs/2005.08866},
    publisher   = {ACL},
}

@inproceedings{larson-etal-2019-evaluation,
    title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    author = "Larson, Stefan  and
      Mahendran, Anish  and
      Peper, Joseph J.  and
      Clarke, Christopher  and
      Lee, Andrew  and
      Hill, Parker  and
      Kummerfeld, Jonathan K.  and
      Leach, Kevin  and
      Laurenzano, Michael A.  and
      Tang, Lingjia  and
      Mars, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-1131"
}

[Budzianowski et al. 2018]
@inproceedings{budzianowski2018large,
    Author = {Budzianowski, Pawe{\l} and Wen, Tsung-Hsien and Tseng, Bo-Hsiang  and Casanueva, I{\~n}igo and Ultes Stefan and Ramadan Osman and Ga{\v{s}}i\'c, Milica},
    title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2018}
}

[Ramadan et al. 2018]
@inproceedings{ramadan2018large,
  title={Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing},
  author={Ramadan, Osman and Budzianowski, Pawe{\l} and Gasic, Milica},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  volume={2},
  pages={432--437},
  year={2018}
}

[Eric et al. 2019]
@article{eric2019multiwoz,
  title={MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines},
  author={Eric, Mihail and Goel, Rahul and Paul, Shachi and Sethi, Abhishek and Agarwal, Sanchit and Gao, Shuyag and Hakkani-Tur, Dilek},
  journal={arXiv preprint arXiv:1907.01669},
  year={2019}
}

[Zang et al. 2020]
@inproceedings{zang2020multiwoz,
  title={MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines},
  author={Zang, Xiaoxue and Rastogi, Abhinav and Sunkara, Srinivas and Gupta, Raghav and Zhang, Jianguo and Chen, Jindong},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, ACL 2020},
  pages={109--117},
  year={2020}
}
```

<!-- CONTACT -->
## Contact

Elaine Lau - https://github.com/yunglau - tsoi.lau@mail.mcgill.ca
