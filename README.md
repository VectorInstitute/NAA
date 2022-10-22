# Neural Agent Assistant framework for Customer Service Support
Neural Agent Assistant framework includes tools and artifacts rooted in deep learning and foundation language models to improve task-oriented customer service conversations and experiences. It is based on the following paper:

Stephen Obadinma, ..., [Bringing the State-of-the-Art to Customers: A Neural Agent Assistant Framework for Customer Service Suppor](ref), EMNLP.

The framework features three high-level components: (1) Intent Identification, (2) Context Retrieval, and (3) Response Generation. Though designed as parts in a pipeline in generating a natural language response to customer questions, the output from any given component can nonetheless be transferred back to the human agent when needed.

<p align="center">
  <a href="https://github.com/VectorInstitute/NAA/blob/main/naa.jpg">
    <img src="https://github.com/VectorInstitute/NAA/blob/main/naa.jpg" alt="pipeline" width="500" height="300">
  </a>
</p>
<p align="center">
  Figure 1. Neural Agent Assistant framework
</p>

# Installing dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

# using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
