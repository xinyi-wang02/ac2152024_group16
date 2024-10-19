## Milestone 2 

#### Project Milestone 2 Organization

```
├── README.md
├── data # DO NOT UPLOAD DATA TO GITHUB, only .gitkeep to keep the directory or a really small sample
├── notebooks
│   └── eda_notebook.ipynb
│   └── model_finetune_experiment.ipynb
├── references
├── reports
│   └── Statement of Work_Sample.pdf
└── src
    ├── datapipeline
    │   ├── data_loader.py
    |   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── preprocess.py
    │   ├── entrypoint.sh
    └── data-versioning
    |   ├── car_preprocessed_dataset.dvc
    |   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── README.md
    └── models
        ├── Dockerfile
        ├── requirements.txt
        ├── vm_instance_screenshot.png
        └── train_model.py
```

# AC215 - Milestone2 - CarsAI

**Team Members**
Nuoya Jiang, John Jun, Seshu Mallina, Harper Wang

**Group Name**
The Car Fever Group (Group 16)

**Project**
This project aims to develop a deep learning-powered vehicle classification system that can identify a car’s make, model, and year from an uploaded image. The system will use deep learning techniques to accurately process the image and display the vehicle’s information to the user. Additionally, we will integrate a chatbot powered by the Gemini API, enabling users to ask questions about the identified car’s features, history, or specifications.

### Milestone2 ###

In this milestone, we have the components for data management, including versioning, as well as the classification model scripts.

**Data**
We used the Stanford Car dataset that has 16,191 car images representing 195 different models of cars. We have stored it in a private Google Cloud Bucket.

**Data Pipeline Containers**
1. One container upload the dataset to a bucket and it also processes the dataset by re-labeling the images for later model train-test split and storing them back to Google Cloud Storage (GCS).

	**Input:** Source and destination GCS locations (bucket name), and required secrets (provided via Docker).

	**Output:** 1. Raw dataset and 2. preprocessed images stored in the specified GCS location.

## Data Pipeline Overview

1. **`ac2152024_group16/data_pipeline/data_loader.py`**
   upload a dataset to GCS. User can specify which dataset to upload. Here we uploaded two datasets: raw and the preprocessed.

2. **`ac2152024_group16/data_pipeline/preprocess.py`**
   This script prepares the data for easier train-test split and generated an annotation file in csv and a mapping dictionary for label encoding in csv.

3. **`ac2152024_group16/data_pipeline/Pipfile`**
   We used the following packages to help with preprocessing:
   - `pandas`
   - `sklearn.preprocessing`

4. **`ac2152024_group16/data_pipeline/Dockerfile`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.


## Running Dockerfile
Instructions for running the Dockerfile can be added here.
To run Dockerfile - `docker-shell.sh`
User can modify `entrypoint.sh` to change parameters

**Models scripts**
- The model folder contains a model fine-tuning script, Dockerfile and package requirements. By the end of this milestone, We are working on running the model train process on a virtual machine instance with CPU. We will set up the model fine-tuning workflow on a virtual machine instance with GPU in future work. 
- The screenshot of Virtual Machine Instance was produced from a CPU-based machine

**Notebooks/Reports**
This folder contains code that is not part of container - EDA, experiment model fine-tuning notebook, and statement of work.
