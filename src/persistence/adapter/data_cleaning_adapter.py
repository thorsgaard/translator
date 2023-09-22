import string

import numpy as np
from pathlib import Path

from src.aplication.ports.secondary.data_cleaning_port import DataCleaningPort


class EngDaDataCleaningAdapter(DataCleaningPort):
    RAW_DATA_URL = Path(__file__).parent.parent.joinpath("raw_data", "dan.txt").resolve()

    def get_clean_data(self) -> np.ndarray:
        """
        This function is interface to the functionality that return clean data to the app
        :return: an array of pair data in respectively primary language and secondary language
        """
        raw_data = self._read_text(self.RAW_DATA_URL)
        eng_da = self._text_to_lines(raw_data)
        return self._clean_data(eng_da)

    def _read_text(self, filename: Path) -> str:
        """
        Open the data file with translations, read the file and close it again
        """
        file = open(filename, mode='rt', encoding='utf-8')

        # read all text
        text = file.read()
        file.close()
        return text

    def _text_to_lines(self, text: str) -> np.ndarray:
        """
        Given a string of translation format it into an array of translation pairs
        """
        translations = text.strip().split('\n')
        translations = np.array([i.split('\t') for i in translations])
        return translations

    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """
        Remove punctuations and convert to lowercase
        """
        data[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 0]]
        data[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 1]]

        for i in range(len(data)):
            data[i, 0] = data[i, 0].lower()
            data[i, 1] = data[i, 1].lower()

        return data
