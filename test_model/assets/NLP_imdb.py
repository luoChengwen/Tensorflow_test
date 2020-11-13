import sys,glob
import numpy as np
sys.path.insert(0,'/anaconda3/envs/Tensorflow_test/lib/python3.7/site-packages/')
import  tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


imdb, info = tfds.load('imdb_reviews', with_info=True , as_supervised=True)
train,test = imdb['train'],imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s, l in train:
	training_sentences.append(s.numpy().decode('utf8'))
	training_labels.append(l.numpy())

for s, l in test:
	testing_sentences.append(s.numpy().decode('utf8'))
	testing_labels.append(l.numpy())


vocab_size = 10000
oov = 'OOV'
truncate = 'post'
maxlen = 300
embeding_dim = 10
output = []

tokenizer = Tokenizer(oov_token=oov, num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)
train_seq = tokenizer.texts_to_sequences(training_sentences)
train_pad = pad_sequences(train_seq, maxlen=maxlen, truncating=truncate)
test_seq = tokenizer.texts_to_sequences(testing_sentences)
test_pad = pad_sequences(test_seq, maxlen=maxlen, truncating=truncate)

def check_sentences(token_s):
	return ' '.join([tokenizer.index_word.get(i) for i in token_s])
	
print(np.array(train_seq[0]).reshape(-1,))
print(check_sentences(train_seq[0]))

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embeding_dim, input_length=maxlen),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(6, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

model.fit(x=train_pad, y=np.array(training_labels), epochs=10,
          validation_data= (test_pad,np.array(testing_labels)))

output.append([ vocab_size, maxlen, embeding_dim,
                model.history.history.get('accuracy')[-1],
                model.history.history.get('val_accuracy')[-1]])

print(np.shape(model.weights[0]))
import io

outvec = io.open('vecs.tsv','w', encoding='utf-8')
outmeta = io.open('meta.tsv','w', encoding='utf-8')
weights = model.layers[0].get_weights()[0]


out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

# for word_num in range(1, vocab_size):
#   word = reverse_word_index[word_num]
#   imbedding = weights[word_num]
#   out_m.write(word + "\n")
#   out_v.write('\t'.join([str(x) for x in imbedding]) + "\n")
  
  
for i in range(1,vocab_size):
	word = tokenizer.index_word[i]
	imbedding = weights[i]
	outmeta.write(word + '\n')
	outvec.write('\t'.join([str(x) for x in imbedding])+'\n')
	print(i,word,outvec)
	