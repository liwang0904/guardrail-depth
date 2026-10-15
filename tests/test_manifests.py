from guardrail_depth.data import load_manifest


def test_manifest_sizes_and_splits():
    expected = {"memorized_sd1": 500, "unmemorized_laion": 500, "memorized_sd2": 219,
                "copycat_characters": 50, "artists": 25, "style_prompts": 250}
    for name, n in expected.items():
        df = load_manifest(name)
        assert len(df) == n, name
        if "split" in df:
            assert set(df["split"]) == {"dev", "test"}, name


def test_memorized_attributes():
    m = load_manifest("memorized_sd1")
    assert set(m["mem_type"]) == {"VM", "TM"}
    # verbatim-memorized prompts reproduce their training image far more often (NeMo SSCD stats)
    med = m.groupby("mem_type")["ref_mean_sscd"].median()
    assert med["VM"] > 0.5 > med["TM"]
    assert (m["n_clip_tokens"] <= 75).all()


def test_copycat_columns():
    c = load_manifest("copycat_characters")
    for col in ["desc60", "kw_lm5", "kw_laion5", "kw_embedding5", "key_entity", "origin_region"]:
        assert (c[col] != "").all(), col
    assert "cartoon" not in " ".join(c["kw_laion5"]).split(", ")
    assert c.loc[c.target == "Link", "gpt4_rewrite_name"].item() == ""  # missing in the CopyCat release


def test_artist_groups():
    a = load_manifest("artists")
    assert a.groupby("group").size().to_dict() == {"deceased_copyright": 5, "living": 10, "public_domain": 10}
    pd_rows = a[a.group == "public_domain"]
    assert (pd_rows["wikiart_label"] != "").all()
