# [CAPTURE-TESTS_HISTORY_LOGS]
from datetime import datetime, timezone

def _fake_rows():
    return [
        {
            "symbol": "ABC",
            "address": "SoMePair111",
            "_score": 55.5,
            "m5": 4.2,
            "liq": 1200,
            "buy_pressure": 60.0,
            "vol_liq_ratio": 8.0,
            "tx_total": 15,
            "buys": 9,
            "sells": 6,
            "age": 12.0,
            "fdv": 10000,
            "url": "https://dexscreener.com/solana/SoMePair111",
        },
        {
            "symbol": "XYZ",
            "address": "SoMePair222",
            "_score": 48.0,
            "m5": 2.1,
            "liq": 800,
            "buy_pressure": 52.0,
            "vol_liq_ratio": 5.5,
            "tx_total": 9,
            "buys": 5,
            "sells": 4,
            "age": 8.0,
            "fdv": 7000,
            "url": "https://dexscreener.com/solana/SoMePair222",
        },
    ]

def test_save_history_and_list(db_module, userA):
    n_saved, msg = db_module.save_scan_rows(userA.id, _fake_rows())
    assert n_saved >= 2

    hist = db_module.list_history(userA.id, limit=10) if "list_history" in dir(db_module) else db_module.list_history(userA.id)
    assert len(hist) >= 2

def test_log_action_is_timestamped(db_module, userA):
    db_module.log_action(userA.id, "login")
    db_module.log_action(userA.id, "scan", {"count": 2})

    # On relit brut via API publique si dispo (sinon, on se contente de ne pas lever d'erreur)
    if hasattr(db_module, "list_logs"):
        logs = db_module.list_logs(userA.id)
        assert len(logs) >= 2
        # timestamps UTC-aware (pas de utcnow())
        for lg in logs:
            ts = getattr(lg, "created_at", None) or getattr(lg, "ts", None)
            if isinstance(ts, datetime):
                assert ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) == timezone.utc.utcoffset(ts)
