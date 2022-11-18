1. Running the notebook in the ml_notebook folder will run the finetuning pipeline for BERT sentiment analysis using the given dataset as augmentation
2. It will also run the evaluation script and build the AUC curve on the train and test split
3. Once the training is done the model will be deposited into the model/ directory whereby the flask app can be run and the enpoint can be tested using postman using the following api route: localhost:5000/predict_sentiment
4. Example input into the ap request is: {
    "text": "btc is the best, and cardano seems to be good"
}