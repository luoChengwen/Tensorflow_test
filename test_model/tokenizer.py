import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing ds?'
]

token = Tokenizer(num_words=100, oov_token='<OOV>')
token.fit_on_texts(sentences)
print(token.index_word)

sequences = token.texts_to_sequences(sentences)
pad_sequences(sequences, maxlen=6, padding='pre', truncating='pre')