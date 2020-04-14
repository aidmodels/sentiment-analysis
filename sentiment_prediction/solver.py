from mlpm.solver import Solver
import pickle

MAX_LEN = 140
class SentimentSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        from keras.models import load_model
        # Do you Init Work here
        self.model = load_model("./pretrained/model-023-0.950231-0.952065.h5")
        with open('./pretrained/tokenizer.pickle', 'rb') as handle:
            self.loaded_tokenizer = pickle.load(handle)
        self.ready()
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        print(data['text'])
        sequences = self.loaded_tokenizer.texts_to_sequences([data['text']])
        print(sequences)
        padding = pad_sequences(sequences, maxlen=MAX_LEN)
        print(padding)
        result = self.model.predict(padding)
        print(result)
        return result # return a dict