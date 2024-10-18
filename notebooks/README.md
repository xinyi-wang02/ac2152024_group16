Notebook Summary:

In the first notebook eda_notebook.ipynb, I focused on exploratory data analysis (EDA). I analyzed the dataset to find patterns with manufacturers, models, and years. I also computed descriptive statistics 
on image sizes and class distributions. From the EDA I was able to note that there is a class imbalance and this will be need to be taken into consideration when building the models. I also tested out 
the bounding box to make sure the numbers given did encapsulate the cars. 

In the second notebook model_finetune_experiment.ipynb, I focused on experimenting with different models. I first did some preliminary data preprocessing to combine the images with their corresponding labels
and other information. I first check to see if the dataset was in the proper form before I trained any models on it. Then I computed some basic statistics on the data. I transformed the dataset into dataloaders
which can be inputted into pytorch models. I also took prebuilt functions from tutorial 12 to save the models. The baseline model that our group chose was resnet18. We also chose to test out vgg16 and mobilenet_v2. 
We found that the vgg16 model performed the best in terms of our metric accuracy, however, it did take the longest to train. Since vgg16 performed the best we decided to finetine the model, however, after finetuning 
its accuracy went down.

The results from the models are shown below: 

| Model Name | Accuracy | Epochs Trained | Notes                             |
|------------|----------|----------------|-----------------------------------|
| Resnet 18 (baseline)    | 11.66%    | 5             | Resnet from pytorch       |
| VGG16                   | 18.42%    | 3             | vgg16 from pytorch        |
| mobilenet_v2            | 10.66%    | 3             | mobilenet_v2 from pytorch |
| VGG16 Finetunes         | 14.26%    | 1             | vgg16 finetuned           |

For the next milestone:

I plan to experiment a bit more with the models to get to an accuracy of >70%. 

Some things I was considering are:
1. Try to train the models for more epochs ie at least 20-25
2. Try ensembling methods to combine predictions from multiple models
3. Also try out other model architecures like GhostNET and ShuffleNET2
4. Finetune other model using different hyperparameters
5. Try out other loss functions other than CrossEntropy()
6. Try SMOTE and other methods to account for class imbalance and not use the Cross Entropy Loss function



