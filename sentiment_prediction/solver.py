from mlpm.solver import Solver
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN=140

class SentimentSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        with open('./pretrained/tokenizer.pickle', 'rb') as handle:
            self.loaded_tokenizer = pickle.load(handle)
        self.model = tf.keras.models.load_model("./pretrained/model.h5")
        self.ready()

    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        sequences = self.loaded_tokenizer.texts_to_sequences([data['input']])
        padding = pad_sequences(sequences, maxlen=MAX_LEN)
        result = self.model.predict(padding, batch_size=1, verbose=1)
        return {"output": result.tolist()} # return a dict
