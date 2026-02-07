# [CAPTURE-TESTS_PRESETS_WATCHLIST]
import json

def test_save_and_list_preset_isolated(db_module, userA, userB):
    params = {"score_min": 40, "liq_min": 500, "age_max": 60}
    ok, msg = db_module.save_preset(userA.id, "scalp-1", json.dumps(params))
    assert ok is True

    a_presets = db_module.list_presets(userA.id)
    b_presets = db_module.list_presets(userB.id)

    assert any(getattr(p, "name", getattr(p, "preset_name", "")) == "scalp-1" for p in a_presets)
    assert not any(getattr(p, "name", getattr(p, "preset_name", "")) == "scalp-1" for p in b_presets)

def test_add_and_list_watchlist(db_module, userA, userB):
    # Ajoute deux pairs pour A
    ok1, _ = db_module.add_watchlist(userA.id, "SoMePair111", "FOO")
    ok2, _ = db_module.add_watchlist(userA.id, "SoMePair222", "BAR")
    assert ok1 is True and ok2 is True

    # Tentative doublon -> doit être gérée (ok False ou message spécifique)
    ok_dup, msg_dup = db_module.add_watchlist(userA.id, "SoMePair111", "FOO")
    assert ok_dup is False

    wlA = db_module.list_watchlist(userA.id)
    wlB = db_module.list_watchlist(userB.id)

    # Vérifie contenu pour A
    pairsA = {(getattr(w, "pair", None) or getattr(w, "pair_address", None)) for w in wlA}
    assert "SoMePair111" in pairsA and "SoMePair222" in pairsA

    # B ne récupère rien (isolation)
    assert len(wlB) == 0
