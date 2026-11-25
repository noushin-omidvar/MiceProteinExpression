import pytest

from mice_protein.data import aggregate_animals, animal_id, load_data, protein_columns


def test_animal_id_parsing():
    assert animal_id("309_1") == "309"
    assert animal_id("50810_15") == "50810"
    with pytest.raises(ValueError):
        animal_id("309")


def test_raw_data_contract():
    df = load_data("Data/Raw_Data/Data_Cortex_Nuclear.xls")
    assert df.shape == (1080, 83)
    assert df.AnimalID.nunique() == 72
    assert len(protein_columns(df)) == 77
    assert df.groupby("AnimalID").size().eq(15).all()


def test_mouse_aggregation_preserves_labels():
    df = load_data("Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(df)
    assert len(mice) == 72
    assert mice["class"].notna().all()
