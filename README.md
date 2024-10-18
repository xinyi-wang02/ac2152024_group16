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

# AC215 - Milestone2 - Cheesy App

**Team Members**
Nuoya Jiang, John Jun, Seshu Mallina, Harper Wang

**Group Name**
The Car Fever Group (Group 16)

**Project**
This project aims to develop a deep learning-powered vehicle classification system that can identify a car’s make, model, and year from an uploaded image. The system will use deep learning techniques to accurately process the image and display the vehicle’s information to the user. Additionally, we will integrate a chatbot powered by the Gemini API, enabling users to ask questions about the identified car’s features, history, or specifications.

### Milestone2 ###

In this milestone, we have the components for data management, including versioning, as well as the classification model scripts.

**Data**
We used the Stanford Car dataset that has 107,291 car images representing 195 different models of cars. We have stored it in a private Google Cloud Bucket.

**Data Pipeline Containers**
1. One container processes the 100GB dataset by resizing the images and storing them back to Google Cloud Storage (GCS).

	**Input:** Source and destination GCS locations, resizing parameters, and required secrets (provided via Docker).

	**Output:** Resized images stored in the specified GCS location.

2. Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.

## Data Pipeline Overview

1. **`src/datapipeline/preprocess_cv.py`**
   This script handles preprocessing on our 100GB dataset. It reduces the image sizes to 128x128 (a parameter that can be changed later) to enable faster iteration during processing. The preprocessed dataset is now reduced to 10GB and stored on GCS.

2. **`src/datapipeline/preprocess_rag.py`**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).

3. **`src/datapipeline/Pipfile`**
   We used the following packages to help with preprocessing:
   - `special cheese package`

4. **`src/preprocessing/Dockerfile(s)`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.


## Running Dockerfile
Instructions for running the Dockerfile can be added here.
To run Dockerfile - `Instructions here`

**Models scripts**
- The model folder contains a model fine-tuning script, Dockerfile and package requirements. By the end of this milestone, We are working on running the model train process on a virtual machine instance with CPU. We will set up the model fine-tuning workflow on a virtual machine instance with GPU in future work. 
- The screenshot of Virtual Machine Instance was produced from a CPU-based machine

**Notebooks/Reports**
This folder contains code that is not part of container - EDA, experiment model fine-tuning notebook, and statement of work.
