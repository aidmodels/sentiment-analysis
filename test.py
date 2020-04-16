import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

MAX_LEN=140

with open('./pretrained/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


model = load_model("./pretrained/model.h5")

data={
    'text':"hello"
}
print(data['text'])
sequences = loaded_tokenizer.texts_to_sequences([data['text']])
padding = pad_sequences(sequences, maxlen=MAX_LEN)
print(padding)
print(model)
result = model.predict(padding)
print(result)