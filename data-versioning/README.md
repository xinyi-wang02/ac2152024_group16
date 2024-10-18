# Data Versioning 

In this folder, we will perform data versioning using dvc for the car identification dataset. Everything will be run inside containers using Docker.

Note that the processed data inside the cs215_car_dataset_w_class/car_preprocessed_folder/all_images will be versioned. 

`docker-shell.sh` file builds and runs a Docker container with access to necessary secrets, Google Cloud credentials, and local configurations, enabling privileged operations and data versioning for a project within a Google Cloud environment.

`docker-entrypoint.sh` authenticates with Google Cloud, mounts a Google Cloud Storage (GCS) bucket to a local directory using gcsfuse, checks if the mount was successful, and then binds the mounted GCS folder to a local directory for use in the application environment, before launching a Python environment using pipenv.

For the purposes of our project, there is no need for frequent updating of our data and, therefore, does not require sophisticated data versioning technique, particularly given the complexity it adds to the project's workflow. 

It would be more helpful to use Vertex AI for the future milestones. 

