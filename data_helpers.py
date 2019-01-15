
import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf

# tf.flags.DEFINE_string("label1_datas", "./train_label1_data_with_slot", "Data source for the label1 data.")
# tf.flags.DEFINE_string("label2_datas", "./train_label2_data_with_slot", "Data source for the label2 data.")
# tf.flags.DEFINE_string("label3_datas", "./train_label3_data_with_slot", "Data source for the label3 data.")

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	# string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	# string = re.sub(r"\'s", " \'s", string)
	# string = re.sub(r"\'ve", " \'ve", string)
	# string = re.sub(r"n\'t", " n\'t", string)
	# string = re.sub(r"\'re", " \'re", string)
	# string = re.sub(r"\'d", " \'d", string)
	# string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", "", string)
	string = re.sub(r"？", "", string)
	string = re.sub(r"！", "", string)
	string = re.sub(r"，", "", string)
	string = re.sub(r"。", "", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def load_data_and_labels(data):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	relation_type = open('./final_2/relation.txt','r',encoding='utf-8').readlines()
	dic_relation = dict()
	num = 0
	for i in relation_type:
		if dic_relation.get(i.split(' ')[0]) is None:
			dic_relation[i.split(' ')[0]] = num
			num += 1

	label1_examples = list(open(data, "r", encoding="utf-8").readlines())
	label1_examples = [s.strip() for s in label1_examples]
	x_text = []
	y = []
	# Split by words
	for i in label1_examples:
		label1_example = i.split('###')
		x_text.append(label1_example[0:-1])
		# print(label1_example[0:-1])
		# x_text = [clean_str(sent) for sent in x_text]

		for key in dic_relation.keys():
			# print(label1_example[-1])
			if key == label1_example[-1]:
				# print(dic_relation[key])
				# print(y.shape)
				# y = np.concatenate([y, dic_relation[key]], axis=0)
				example = [0]*24
				example[dic_relation[key]] = 1
				y.append(example)

	return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

# load_data_and_labels(FLAGS.label1_datas, FLAGS.label2_datas, FLAGS.label3_datas)