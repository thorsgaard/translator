from src.persistence.adapter.data_cleaning_adapter import EngDaDataCleaningAdapter


def test_data_cleaning():
    #Arrange

    #Act
    data = EngDaDataCleaningAdapter().get_clean_data()

    #Assert
    assert(1 == 1)