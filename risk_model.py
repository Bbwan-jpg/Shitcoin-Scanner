# risk_model.py
# Utilitaire d'inférence : charge model_pump.pkl et calcule p(risque) pour des pairs DexScreener

import time                          # Gestion de l'heure/temps (pour l'âge du pair)
from typing import List, Dict, Any, Tuple  # Types pour l'annotation
import numpy as np                   # Calcul numérique (log1p, arrays)
import pandas as pd                  # DataFrames pour features / sorties
import joblib                        # Chargement du modèle scikit-learn sérialisé

# ==== les mêmes features basiques que dans pump_learn (version inference-only) ====

def _safe_float(x, default=0.0):
    try: 
        return float(x)              # Convertit en float
    except: 
        return default               # Valeur de secours si conversion impossible

FEATURE_ORDER = [                    # Ordre canonique des features (doit matcher l'entraînement)
    "m5","m15","h1","h6","h24",
    "vol_liq_5","vol_liq_60",
    "buy_pressure","tx_imbalance","tx5",
    "log_liq","log_vol5","log_vol1h","log_fdv","log_price","log_age",
    "age_min", "price", "liq", "fdv", "vol_m5", "vol_h1", "buys", "sells"
]

def features_from_pair(p: Dict[str, Any]) -> Dict[str, float]:
    pc   = p.get("priceChange") or {}             # Sous-objet variations de prix
    vol  = p.get("volume") or {}                  # Sous-objet volumes
    txm5 = (p.get("txns") or {}).get("m5", {}) or {}  # Transactions sur 5 minutes

    m5  = _safe_float(pc.get("m5"))              # % change 5m
    m15 = _safe_float(pc.get("m15", pc.get("m10", 0)))  # % change 15m (fallback 10m si présent)
    h1  = _safe_float(pc.get("h1"))              # % change 1h
    h6  = _safe_float(pc.get("h6"))              # % change 6h
    h24 = _safe_float(pc.get("h24"))             # % change 24h

    liq   = _safe_float((p.get("liquidity") or {}).get("usd"))  # Liquidité USD
    vol5  = _safe_float(vol.get("m5"))           # Volume 5m
    vol1h = _safe_float(vol.get("h1"))           # Volume 1h
    fdv   = _safe_float(p.get("fdv"))            # FDV
    price = _safe_float(p.get("priceUsd"))       # Prix USD

    buys  = int(txm5.get("buys", 0) or 0)        # Nombre d'achats 5m
    sells = int(txm5.get("sells", 0) or 0)       # Nombre de ventes 5m
    tx5   = buys + sells                         # Total transactions 5m

    created_at = int(p.get("pairCreatedAt") or 0)                            # Timestamp création (ms)
    age_min = (time.time()*1000 - created_at)/60000.0 if created_at else 1e9 # Âge en minutes (fallback grand)

    buy_pressure = (buys / tx5) if tx5 > 0 else 0.0        # Part des achats
    vol_liq_5    = (vol5 / liq) if liq > 0 else 0.0        # Volume5 / Liquidité
    vol_liq_60   = (vol1h / liq) if liq > 0 else 0.0       # Volume1h / Liquidité
    tx_imbalance = ((buys - sells) / tx5) if tx5 > 0 else 0.0  # Déséquilibre orderflow

    return {                                              # Dictionnaire des features brutes + log-transformées
        "m5": m5, "m15": m15, "h1": h1, "h6": h6, "h24": h24,
        "vol_liq_5": vol_liq_5, "vol_liq_60": vol_liq_60,
        "buy_pressure": buy_pressure, "tx_imbalance": tx_imbalance, "tx5": tx5,
        "log_liq": np.log1p(liq), "log_vol5": np.log1p(vol5), "log_vol1h": np.log1p(vol1h),
        "log_fdv": np.log1p(max(fdv, 0.0)), "log_price": np.log1p(max(price, 0.0)),
        "log_age": np.log1p(max(age_min, 0.0)),
        "age_min": age_min, "price": price, "liq": liq, "fdv": fdv,
        "vol_m5": vol5, "vol_h1": vol1h, "buys": buys, "sells": sells
    }

def df_from_pairs(pairs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, meta = [], []                              # Accumulateurs features / métadonnées
    for p in pairs:                                   # Parcourt chaque pair DexScreener
        try:
            f = features_from_pair(p)                 # Extrait les features
            rows.append([f.get(k, 0.0) for k in FEATURE_ORDER])  # Range selon l'ordre canonique
            meta.append({                             # Métadonnées parallèles (pour restitution)
                "pair": p.get("pairAddress",""),
                "symbol": (p.get("baseToken") or {}).get("symbol","?"),
                "url": f"https://dexscreener.com/solana/{p.get('pairAddress','')}"
            })
        except Exception:
            continue                                  # Ignore les pairs malformés
    X = pd.DataFrame(rows, columns=FEATURE_ORDER)     # Matrice X de features
    M = pd.DataFrame(meta)                            # DF des métadonnées
    return X, M

# ==== API publique ====

def load_model(model_path: str):
    """Retourne le pack {model: Pipeline, features: list}"""
    return joblib.load(model_path)                    # Charge le pkl entraîné (pipeline + liste de features)

def predict_pairs(pairs: List[Dict[str, Any]], pack) -> pd.DataFrame:
    """Renvoie un DF {pair,symbol,url,prob_risk} ordonné par proba décroissante."""
    if not pairs:
        return pd.DataFrame(columns=["pair","symbol","url","prob_risk"])  # Cas vide

    pipe = pack["model"]                              # Récupère le Pipeline scikit-learn
    feats = pack["features"]                          # Ordre des features attendu par le modèle

    X, M = df_from_pairs(pairs)                       # Construit X (features) et M (meta)
    if X.empty:
        return pd.DataFrame(columns=["pair","symbol","url","prob_risk"])  # Rien à prédire

    X = X.reindex(columns=feats, fill_value=0.0)      # Aligne colonnes sur l'ordre attendu
    probs = pipe.predict_proba(X.values)[:, 1]        # Probabilité classe "risque" (colonne 1)
    M["prob_risk"] = probs                            # Ajoute la proba au DF meta
    return M.sort_values("prob_risk", ascending=False).reset_index(drop=True)  # Trie décroissant

def explain_one(pair: Dict[str, Any], pack, top: int = 8) -> pd.DataFrame:
    """
    Approx des contributions: coef * (feature_standardisée).
    Retourne top features (nom, contribution, valeur brute).
    """
    pipe = pack["model"]                              # Pipeline entraîné
    feats = pack["features"]                          # Liste des features
    clf = pipe.named_steps["clf"]                     # Classifieur (LogisticRegression)
    scaler = pipe.named_steps["scaler"]               # StandardScaler

    f = features_from_pair(pair)                      # Features brutes pour ce pair
    x = np.array([[f.get(k, 0.0) for k in feats]])    # Vecteur x (1, n_features) dans l'ordre
    xs = scaler.transform(x)                          # Standardisation comme à l'entraînement
    contrib = clf.coef_.ravel() * xs.ravel()          # Contribution au log-odds: coef * feature_std

    df = pd.DataFrame({"feature": feats, "contrib": contrib, "value": x.ravel()})  # Tableau explicatif
    df["abs"] = df["contrib"].abs()                   # Magnitude pour trier
    df = df.sort_values("abs", ascending=False).drop(columns=["abs"])  # Tri par importance
    return df.head(top).reset_index(drop=True)        # Garde les top n et réindexe
