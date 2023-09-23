from typing import List

import numpy as np
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


class EncodingService:

    def __init__(self, tokens: List[str]) -> None:
        self.tokenizer = self.compute_tokenizer(tokens)

    def encode_tokens(self,
                      length: int,
                      tokens: List[str]) -> np.ndarray[int]:
        """
        Responsible to encode tokens into a vector of integers using Kera's
        and padding the sequences with 0's to the desired length
        """
        seq = self.tokenizer.texts_to_sequences(tokens)
        seq = pad_sequences(seq, maxlen=length, padding='post')
        return seq

    def compute_tokenizer(self, tokens: List[str]) -> Tokenizer:
        """
        Compute the tokens using Keras package
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokens)
        return tokenizer
