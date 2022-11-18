1. Running the notebook in the ml_notebook folder will run the finetuning pipeline for BERT sentiment analysis using the given dataset as augmentation
2. It will also run the evaluation script and build the AUC curve on the train and test split
3. Once the training is done the model will be deposited into the model/ directory whereby the flask app can be run and the enpoint can be tested using postman using the following api route: localhost:5000/predict_sentiment
4. Example input into the api request is: {
    "text": "btc is the best, and cardano seems to be good"
}
5. There is a script called sentiment_inference_test.py that will process some examples and output the inference time as well
6. Additional Notes: The model seems to be good at handling explicit sentiment but when it comes to constructive statements like for example "the btc ecosystem needs improvement" tend to be mis classified. There needs to be more data in the vicinity of statements that are not so explicit to improve the models robustness
7. While discerning explicit binarized sentiment is valuable, gathering sentiment on more nuanced discourse can provide insight at a deeper level and can be a huge value add
