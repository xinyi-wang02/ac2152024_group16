Command to run to cloud build and deploy:  
gcloud builds submit --tag gcr.io/my-test-project-215/index --project=my-test-project-215 
gcloud run deploy --image gcr.io/my-test-project-215/index --platform managed --project=my-test-project-215 --memory=1Gi

For milestone 3 the model is currently deployed on GCP cloud run. However, when requests are run there is usually a 500 error. For future milestones, this will be fixed. When the code is deployed locally and the requests are run using the test.py function the code gives out the prediction from the model. 
