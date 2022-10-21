# Neural Agent Assistant framework for Customer Service Support
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

# Data 
You can upload data from <a href="https://drive.google.com/drive/folders/1Kp7yYPqAfBFZ4ff7YCV7c4RwUSPkamdj?usp=sharing"> here. 

# using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
