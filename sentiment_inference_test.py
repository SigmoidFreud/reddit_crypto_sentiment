import pathlib
import time
from pprint import pprint

import requests
import json


# call get service with headers and params
def sentiment_test(endpoint='predict_sentiment', url='http://127.0.0.1:5000'):
	sentiment_test_comment_text = ['btc is a great replacement for money',
	                               'cardano is a shitcoin, it has no place in the future ecosystem',
	                               'btc needs improvement to its overall ecosystem',
	                               'btc is about to go to the moon',
	                               'ftx just blew up into oblivion']
	
	output_list = []
	url = f'{url}/{endpoint}'
	for text in sentiment_test_comment_text:
		feature_data_dict = {'text': text}
		# print(feature_data)
		start = time.time()
		
		response = requests.post(url, json=feature_data_dict)
		end = time.time()
		elapsed_time = end - start
		output_json = response.json()
		output_json['comment_text'] = text
		output_json['inference_time'] = elapsed_time
		
		output_list.append(output_json)
	pprint(output_list)
	return output_list


sentiment_test()
