import numpy as np
import pandera.pandas as pa
import pytest

from resonance_pipeline.schema import data_dictionary_md
from resonance_pipeline.validate import validate_clean, validate_raw


def test_valid_frame_passes(raw_df):
    assert len(validate_raw(raw_df)) == len(raw_df)


def test_null_in_required_feature_fails(raw_df):
    raw_df.loc[3, "energy"] = np.nan
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(raw_df)


def test_popularity_out_of_range_fails(raw_df):
    raw_df.loc[0, "popularity"] = 101
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(raw_df)


def test_missing_column_fails(raw_df):
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(raw_df.drop(columns=["valence"]))


def test_clean_schema_rejects_tempo_outlier(clean_df):
    clean_df.loc[0, "tempo"] = 0.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_clean(clean_df)


def test_dictionary_lists_every_feature():
    md = data_dictionary_md()
    assert "`danceability`" in md and "`track_genre`" in md


def test_dictionary_ranges_are_human_readable():
    md = data_dictionary_md()
    assert "0–1" in md and "≥ 30" in md and "{0, 1}" in md and "popularity_percentile_within_genre" in md
    assert "min_value" not in md
