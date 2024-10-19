# Data Versioning 

In our project, we use a curated dataset, Stanford Car Dataset, which is a well-established dataset of ~16,000k car images and labels. While we believe that the original dataset is reliable and does not require constant upkeep, we do find data versioning infrastructure a helpful practice to accommodate cases of addition of new images to the original dataset or changes to the preprocessed data with an improved preprocessing method.

Hence, we employ dvc, which tracks and manages changes in large datasets, similar to how Git tracks code. DVC stores data files in external storage (like cloud services) while keeping lightweight references in the Git repository. This enables us to version datasets, share them with collaborators, and ensure reproducibility in machine learning experiments without storing large files in Git itself.

We have the containerized data versioning strategy to ensure that it is reproducible and portable. The version control history is shown below:

`docker-shell.sh` file builds and runs a Docker container with access to necessary secrets, Google Cloud credentials, and local configurations, enabling privileged operations and data versioning for a project within a Google Cloud environment.

`docker-entrypoint.sh` authenticates with Google Cloud, mounts a Google Cloud Storage (GCS) bucket to a local directory using gcsfuse, checks if the mount was successful, and then binds the mounted GCS folder to a local directory for use in the application environment, before launching a Python environment using pipenv.

Terminal Screenshot (dvc add command)
![Screenshot 2024-10-18 at 8 09 39 PM](https://github.com/user-attachments/assets/f259ba96-d874-4b2b-803e-121d42fb4323)

GCS Screenshot (dvc_store folder)
![Screenshot 2024-10-18 at 8 12 16 PM](https://github.com/user-attachments/assets/052ddc0b-9b54-438c-86ef-5c76be18ef60)
