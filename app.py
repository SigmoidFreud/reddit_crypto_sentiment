import json
import spacy
import torch
from flask import Flask, request, jsonify
from nltk import PorterStemmer
from torch import nn
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

lm = spacy.load("en_core_web_sm")


class BertClassifier(nn.Module):
	"""Bert Model for Classification Tasks.
	"""
	
	def __init__(self, freeze_bert=False):
		"""
		@param    bert: a BertModel object
		@param    classifier: a torch.nn.Module classifier
		@param    freeze_bert (bool): Set `False` to fine-tune the BERT model
		"""
		super(BertClassifier, self).__init__()
		# Specify hidden size of BERT, hidden size of our classifier, and number of labels
		D_in, H, D_out = 768, 50, 2
		
		# Instantiate BERT model
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		
		# Instantiate an one-layer feed-forward classifier
		self.classifier = nn.Sequential(
			nn.Linear(D_in, H),
			nn.ReLU(),
			# nn.Dropout(0.5),
			nn.Linear(H, D_out)
		)
		
		# Freeze the BERT model
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False
	
	def forward(self, input_ids, attention_mask):
		"""
		Feed input to BERT and the classifier to compute logits.
		@param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
					  max_length)
		@param    attention_mask (torch.Tensor): a tensor that hold attention mask
					  information with shape (batch_size, max_length)
		@return   logits (torch.Tensor): an output tensor with shape (batch_size,
					  num_labels)
		"""
		# Feed input to BERT
		outputs = self.bert(input_ids=input_ids,
		                    attention_mask=attention_mask)
		
		# Extract the last hidden state of the token `[CLS]` for classification task
		last_hidden_state_cls = outputs[0][:, 0, :]
		
		# Feed input to classifier to compute logits
		logits = self.classifier(last_hidden_state_cls)
		
		return logits


model = BertClassifier()
model.load_state_dict(torch.load('./model/crypto_finetuned_reddit_sentiment'))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def text_preprocessing(text):
	my_doc = lm(text)
	
	# Create list of word tokens
	token_list = []
	for token in my_doc:
		token_list.append(token.text)
	filtered_sentence = []
	
	for word in token_list:
		lexeme = lm.vocab[word]
		if not lexeme.is_stop:
			filtered_sentence.append(word)
	stemmer = PorterStemmer()
	text = ' '.join([stemmer.stem(y) for y in filtered_sentence])
	return text


# Add the outputs to the lists

def get_pred(sent):
	max_comment_length = 463
	encoded_sent = tokenizer.encode_plus(
		text=text_preprocessing(sent),  # Preprocess sentence
		add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
		max_length=max_comment_length,  # Max length to truncate/pad
		padding='longest',  # Pad sentence to max length
		# return_tensors='pt',           # Return PyTorch tensor
		return_attention_mask=True  # Return attention mask
	)
	input_ids = encoded_sent.get('input_ids')
	attention_masks = encoded_sent.get('attention_mask')
	input_ids = torch.tensor([input_ids])
	attention_masks = torch.tensor([attention_masks])
	with torch.no_grad():
		logits = model(input_ids, attention_masks)
	preds = torch.argmax(logits, dim=1).flatten()
	return preds


@app.route('/predict_sentiment', methods=['POST'])
def predict():  # put application's code here
	data = request.get_json(force=True)
	# print(data)
	sent = data['text']
	result = get_pred(sent).tolist()[0]
	if result:
		sentiment = 'Positive'
	else:
		sentiment = 'Negative'
	pred = {'Sentiment': sentiment}
	# print(pred)
	return pred


if __name__ == '__main__':
	app.run()
