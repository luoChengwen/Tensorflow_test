import tensorflow as tf
import pandas as pd
import json
import numpy as np


with open("/Users/daisy/Downloads/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    
sentences = []
label = []

for i in datastore:
	
	sentences.append(i.get('headline'))
	label.append(i.get('is_sarcastic'))

train_x, test_x = sentences[:20000], sentences[20000:]
train_y, test_y = label[:20000], label[20000:]

oov = 'OOV'
maxlen = 120
vocab_size = 10000
embedding_dim = 16

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov)
tokenizer.fit_on_texts(train_x)
print(tokenizer.index_word)

train_x_seq = tokenizer.texts_to_sequences(train_x)
train_x_seq = tf.keras.preprocessing.sequence.pad_sequences(train_x_seq, padding= 'post', truncating='post', maxlen=maxlen)

train_y = np.array(train_y)
test_x_seq = tokenizer.texts_to_sequences(test_x)
test_x_seq = tf.keras.preprocessing.sequence.pad_sequences(test_x_seq, padding= 'post', truncating='post', maxlen=maxlen)
test_y = np.array(test_y)

print(train_x_seq[0])
print(' '.join([tokenizer.index_word.get(i,'?') for i in train_x_seq[0] ]))

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = maxlen),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(6, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')])


model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

model.fit(x= train_x_seq, y= train_y, epochs=10, validation_data=(test_x_seq, test_y))


