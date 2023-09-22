from abc import ABC, abstractmethod

from numpy import ndarray


class DataCleaningPort(ABC):

    @abstractmethod
    def get_clean_data(self) -> ndarray:
        """
        This function is interface to the functionality that return clean data to the app
        :return: an array of pair data in respectively primary language and secondary language
        """
        pass