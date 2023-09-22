from typing import List

import numpy as np
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


class EncodingService:

    def __init__(self, lines: List[str]) -> None:
        self.tokenizer = self.compute_tokenizer(lines)

    def encode_list(self,
                    length: int,
                    lines: List[str]) -> np.ndarray[int]:
        """
        This method is responsible to encode a string into a vector of integers using Kera's
        and padding the sequences with 0's to the desired length
        """
        seq = self.tokenizer.texts_to_sequences(lines)
        seq = pad_sequences(seq, maxlen=length, padding='post')
        return seq

    def compute_tokenizer(self, lines: List[str]) -> Tokenizer:
        """
        Compute the tokens using Keras package
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
