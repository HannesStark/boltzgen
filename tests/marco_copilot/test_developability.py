from marco_copilot.developability import features


def test_features_basic():
    f = features("ACDEKNNST")
    assert f["length"] == 9
    assert f["cys_count"] == 1
    assert f["nglyc_motifs"] >= 1
