# beedle-p2 Ethical Gender Classification
The goal of this project is to classify images of people by their gender, with ethical concerns in mind.

We used the Project Deon Data Science Ethics Checklist tool to analyze the ethical quality of our gender classification pipeline. 

Our pipeline uses a Google Dataproc server to consume a giant dataset of face images and their associated features as presented in Merler's 2019 Diversity in Faces paper (https://arxiv.org/abs/1901.10436). The features in the dataset are designed to alleviate bias in facial recognition software. This, hand in hand with Deon, should hopefully help us develop an ethical facial recognition pipeline.

# Cluster Configuration:
* 1 master node (n1-standard-4, 4 CPU's,15GB ) and 5 worker nodes (n1-highmen-4, 26GB, 20 CPU's)
* we had to enable component gateway to access the jupyter notebook on the GCP cluster

# To run code:
open src/
