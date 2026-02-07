import os                       # Utilitaires système (fichiers, variables d'env)
import sys                      # (non utilisé ici) – pourrait servir pour argv / path
import time                     # Horodatage et temporisations
import math                     # (non utilisé ici) – fonctions math si besoin
import joblib                   # Sauvegarde/chargement du modèle sklearn
import ccxt                     # Client d’exchanges crypto (Binance ici)
import numpy as np              # Calcul numérique
import pandas as pd             # Manipulation de tableaux/Series
from dateutil import parser as du              # Parsing de dates ISO → datetime
from typing import Dict, Any, List, Optional  # Types d’annotation
from sklearn.pipeline import Pipeline         # Pipeline sklearn (scaler + modèle)
from sklearn.preprocessing import StandardScaler   # Normalisation des features
from sklearn.linear_model import LogisticRegression # Classifieur
from sklearn.metrics import classification_report   # Rapport de performance

# ==============================
# Réglages
# ==============================
DEFAULT_EVENTS_CSV = "pumpdump_labels_sample.csv"   # CSV des événements (train)
DEFAULT_MODEL_PATH = "model_pump.pkl"               # Chemin du modèle sauvegardé
DEFAULT_SYMBOLS_TXT = "symbols.txt"                 # Liste de tickers pour la prédiction

TIMEFRAME = "1m"            # Granularité des bougies pour features
LOOKBACK_MIN = 120          # Fenêtre d’historique en minutes
PAUSE_S = 0.25              # Pause entre appels ccxt (rate limit)

# Négatifs auto si une seule classe
NEG_PER_POS = 2             # Nombre d’échantillons négatifs créés par positif
RANDOM_SEED = 42            # Graine RNG pour reproductibilité

# ==============================
# Helpers généraux
# ==============================
def ok(s: str) -> bool:
    return s is not None and str(s).strip() != ""   # True si chaîne non vide

def to_utc_ms(s: Optional[str]) -> Optional[int]:
    if not ok(s):                                   # Ignore si vide
        return None
    try:
        return int(du.isoparse(s).timestamp() * 1000)  # ISO → epoch ms
    except Exception:
        return None                                  # None si parsing impossible

def choose_market_symbol(exchange: ccxt.Exchange, base: str) -> Optional[str]:
    """
    Essaie base/USDT puis base/BUSD. Retourne None si introuvable.
    """
    base = base.upper().strip()                     # Normalise le ticker
    candidates = [f"{base}/USDT", f"{base}/BUSD"]   # Priorité aux marchés usuels
    markets = exchange.load_markets()               # Charge la liste des marchés
    for m in candidates:
        if m in markets:                            # Retourne le premier existant
            return m
    return None                                     # Aucun marché trouvé

def fetch_ohlcv_safe(exchange: ccxt.Exchange, market: str, timeframe: str, since_ms: int, limit: int = 2000):
    try:
        return exchange.fetch_ohlcv(market, timeframe=timeframe, since=since_ms, limit=limit)  # Télécharge OHLCV
    except Exception:
        return []                                    # En cas d’erreur → liste vide

def ohlcv_to_df(ohlcv: List[List[Any]]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])  # DF vide standardisé
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"]) # Colonnes ccxt classiques
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)      # ms epoch → timestamp UTC
    return df

# ==============================
# Features à partir des OHLCV
# ==============================
def pct_change(series: pd.Series, n: int) -> float:
    if len(series) < n+1:                           # Pas assez de points
        return 0.0
    try:
        a = float(series.iloc[-n-1])                # Valeur n périodes avant
        b = float(series.iloc[-1])                  # Dernière valeur
        return 0.0 if a == 0 else 100.0 * (b - a) / a  # Variation % sur n
    except Exception:
        return 0.0

def rolling_vol(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return 0.0
    return float(series.iloc[-n:].pct_change().std() or 0.0)  # Écart-type des retours sur n

def volume_surge(vol: pd.Series, n_now: int = 1, n_ref: int = 60) -> float:
    """
    Ratio volume(dernier n_now) / médiane(volume des n_ref précédents).
    """
    if len(vol) < max(n_now, n_ref)+1:
        return 0.0
    now = float(vol.iloc[-n_now:].sum())            # Volume récent agrégé
    ref = float(vol.iloc[-(n_ref+1):-1].median() or 0.0)  # Référence historique (médiane)
    return now / ref if ref > 0 else 0.0

def wick_ratio(df: pd.DataFrame) -> float:
    """
    Longueur de mèche (H-L) vs corps (|C-O|) de la dernière bougie.
    """
    if df.empty:
        return 0.0
    o,h,l,c = map(float, df.iloc[-1][["o","h","l","c"]].values)  # Dernière bougie
    body = abs(c-o)                              # Taille du corps
    range_ = max(1e-12, h-l)                     # Amplitude totale (évite /0)
    return (range_ / max(1e-12, body)) if body > 0 else (range_ / 1e-6)  # Ratio mèche/corps

def build_features_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        # vecteur nul
        return {k:0.0 for k in FEATURE_ORDER}    # Retourne toutes features à 0

    close = df["c"].astype(float)                # Série des clôtures
    vol   = df["v"].astype(float)                # Série des volumes

    feats = {
        # momentum
        "ret_5m": pct_change(close, 5),          # Variation 5 min
        "ret_15m": pct_change(close, 15),        # Variation 15 min
        "ret_60m": pct_change(close, 60),        # Variation 60 min
        # volat / microstructure
        "vola_15": rolling_vol(close, 15),       # Volatilité glissante 15
        "vola_60": rolling_vol(close, 60),       # Volatilité glissante 60
        "wick_ratio": wick_ratio(df),            # Ratio mèches / corps
        # volumes
        "vol_surge_1_60": volume_surge(vol, 1, 60),    # Spike vol 1 vs 60
        "vol_surge_5_60": volume_surge(vol, 5, 60),    # Spike vol 5 vs 60
        "vol_surge_15_60": volume_surge(vol, 15, 60),  # Spike vol 15 vs 60
        # niveaux
        "price": float(close.iloc[-1]),          # Dernier prix
        "log_price": float(np.log1p(max(float(close.iloc[-1]), 0.0))),  # log1p(price)
    }
    return feats

# Ordre fixe
FEATURE_ORDER = [
    "ret_5m","ret_15m","ret_60m",
    "vola_15","vola_60","wick_ratio",
    "vol_surge_1_60","vol_surge_5_60","vol_surge_15_60",
    "price","log_price"
]

def features_for_symbol_before(exchange: ccxt.Exchange, base: str, t0_ms: int) -> Optional[Dict[str,float]]:
    """
    Construit les features sur [t0 - LOOKBACK_MIN, t0] (inclus) pour SYMBOL/USDT|BUSD.
    """
    market = choose_market_symbol(exchange, base)   # Choix du marché
    if not market:
        return None
    since = t0_ms - LOOKBACK_MIN * 60 * 1000        # Début de fenêtre d’OHLCV
    raw = fetch_ohlcv_safe(exchange, market, TIMEFRAME, since)  # Télécharge les bougies
    df = ohlcv_to_df(raw)                           # Convertit en DataFrame
    # ne garder que <= t0
    df = df[df["t"].astype(np.int64)//10**6 <= t0_ms]  # Filtre les bougies futures
    return build_features_from_df(df)               # Calcule le vecteur de features

def features_for_symbol_now(exchange: ccxt.Exchange, base: str) -> Optional[Dict[str,float]]:
    """
    Construit les features sur [now - LOOKBACK_MIN, now] pour SYMBOL/USDT|BUSD.
    """
    market = choose_market_symbol(exchange, base)   # Choix du marché
    if not market:
        return None
    now_ms = int(time.time() * 1000)                # Timestamp actuel (ms)
    since = now_ms - LOOKBACK_MIN * 60 * 1000       # Début de fenêtre
    raw = fetch_ohlcv_safe(exchange, market, TIMEFRAME, since)  # Récup OHLCV
    df = ohlcv_to_df(raw)                           # DataFrame OHLCV
    return build_features_from_df(df)               # Vecteur de features

# ==============================
# Dataset d'entraînement (CSV PumpDump)
# ==============================
def load_events_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)                          # Charge le CSV des événements
    # colonnes utilisées :
    # announced_at_utc, pump_date_utc (optionnel), symbol, exchange, label_numeric, is_success
    for col in ["symbol","exchange","announced_at_utc"]:
        if col not in df.columns:                   # Vérifie les colonnes clés
            raise ValueError(f"Colonne manquante : {col}")
    # normaliser
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()   # Tickers propres
    df["exchange"] = df["exchange"].astype(str).str.strip()           # Nom d’exchange propre
    # harmoniser champs de label si présents
    if "label_numeric" in df.columns:
        df["label_numeric"] = pd.to_numeric(df["label_numeric"], errors="coerce").fillna(1).astype(int)  # Nettoie label_numeric
    if "is_success" in df.columns:
        df["is_success"] = pd.to_numeric(df["is_success"], errors="coerce").fillna(0).astype(int)        # Nettoie is_success
    return df

def pick_target_column(ev: pd.DataFrame) -> Optional[str]:
    """
    Choisit la colonne de label :
    1) is_success si binaire et contient 0 & 1
    2) label_numeric si binaire et contient 0 & 1
    3) sinon None (on générera des négatifs synthétiques)
    """
    if "is_success" in ev.columns:
        uniq = ev["is_success"].dropna().astype(int).unique() # Valeurs uniques
        if set(uniq) >= {0,1}:                                 # Contient 0 et 1
            return "is_success"
    if "label_numeric" in ev.columns:
        uniq = ev["label_numeric"].dropna().astype(int).unique()
        if set(uniq) >= {0,1}:
            return "label_numeric"
    return None

def get_binance_universe(exchange: ccxt.Exchange) -> List[str]:
    """
    Renvoie la liste des bases (tickers) disponibles sur /USDT ou /BUSD.
    """
    mkts = exchange.load_markets()                 # Charge marchés Binance
    bases = set()
    for sym in mkts.keys():                        # Itère tous les symboles
        if sym.endswith("/USDT") or sym.endswith("/BUSD"):
            base = sym.split("/")[0]               # Récupère la base
            # on filtre quelques "bases" bizarres
            if len(base) >= 2 and base.isalpha():  # Garde des tickers plausibles
                bases.add(base.upper())
    return sorted(list(bases))                     # Liste triée

def build_training_frame(events_csv: str) -> pd.DataFrame:
    print(f"Chargement des événements : {events_csv}")  # Log
    ev = load_events_csv(events_csv)                # Lit et nettoie le CSV
    ev = ev[ev["exchange"].str.lower()=="binance"].copy()  # Garde Binance uniquement

    # t0 = pump_date_utc si dispo, sinon announced_at_utc
    ev["t0_ms"] = ev["pump_date_utc"].apply(to_utc_ms) if "pump_date_utc" in ev.columns else None  # t0 depuis pump_date_utc
    if "t0_ms" in ev:
        ev["t0_ms"] = ev["t0_ms"].fillna(ev["announced_at_utc"].apply(to_utc_ms))  # fallback announced_at_utc
    else:
        ev["t0_ms"] = ev["announced_at_utc"].apply(to_utc_ms)                      # si pas de colonne précédente

    ev = ev[ev["t0_ms"].notna()].copy()           # Supprime lignes sans t0

    # label de référence
    target_col = pick_target_column(ev)           # Choisit la colonne label
    if target_col:
        print(f"Label utilisé = '{target_col}'")  # Info si label dispo
    else:
        print("Aucun label binaire utilisable trouvé -> génération de négatifs synthétiques…")

    ex = ccxt.binance({"enableRateLimit": True})  # Instancie client ccxt Binance
    rng = np.random.default_rng(RANDOM_SEED)      # RNG pour choix aléatoires

    rows = []                                     # Accumule les vecteurs de features
    labels = []                                   # Accumule les labels
    meta = []                                     # Métadonnées (symbol, t0, kind)

    # ---- Positifs (ou lignes événementielles) ----
    print(f"Construction des features avant t0 (lookback={LOOKBACK_MIN} min) …")
    for _, r in ev.iterrows():                    # Parcourt chaque événement
        base = r["symbol"]                         # Ticker
        t0   = int(r["t0_ms"])                     # Timestamp de référence
        feats = features_for_symbol_before(ex, base, t0)  # Features avant t0
        time.sleep(PAUSE_S)                        # Respecte rate limit
        if feats is None:
            continue                               # Ignore si marché absent
        rows.append([feats.get(k,0.0) for k in FEATURE_ORDER])  # Ajoute features dans l’ordre
        yval = int(r[target_col]) if target_col else 1          # Label positif si pas de colonne
        labels.append(yval)                         # Append label
        meta.append({"symbol": base, "t0": r.get("pump_date_utc") or r.get("announced_at_utc"), "kind":"pos"})  # Meta

    if not rows:
        raise RuntimeError("Aucun feature construit pour les événements. Vérifie les symboles/markets.")  # Sécurité

    # ---- Si une seule classe -> créer des négatifs ----
    unique_classes = set(labels)                   # Classes présentes
    if len(unique_classes) < 2:
        # construire un univers, puis échantillonner pour chaque positif
        uni = get_binance_universe(ex)             # Univers Binance (bases)
        pos_syms = set([m["symbol"] for m in meta])# Symboles déjà utilisés
        pool = [s for s in uni if s not in pos_syms]  # Pool pour négatifs
        if not pool:
            raise RuntimeError("Impossible de créer des négatifs: pool vide.")
        print(f"Génération de négatifs : {NEG_PER_POS} par positif (pool ~{len(pool)} bases)…")

        extra_rows, extra_y, extra_meta = [], [], []
        for m in meta:                              # Pour chaque positif
            t0 = int(to_utc_ms(m["t0"]))           # Même t0
            cand = rng.choice(pool, size=min(NEG_PER_POS, len(pool)), replace=False)  # Échantillonne des bases
            for base in cand:
                feats = features_for_symbol_before(ex, base, t0)  # Features pour une base “négative”
                time.sleep(PAUSE_S)
                if feats is None:
                    continue
                extra_rows.append([feats.get(k,0.0) for k in FEATURE_ORDER])  # Ajoute features
                extra_y.append(0)                                             # Label négatif
                extra_meta.append({"symbol": base, "t0": m["t0"], "kind":"neg"})  # Meta négatif

        print(f"Négatifs générés: {len(extra_rows)}")  # Log
        rows.extend(extra_rows)                        # Concat features
        labels.extend(extra_y)                         # Concat labels
        meta.extend(extra_meta)                        # Concat meta

    # ---- Construire la table finale ----
    X = pd.DataFrame(rows, columns=FEATURE_ORDER)     # Matrice X
    y = pd.Series(labels, name="label")               # Série y
    M = pd.DataFrame(meta)                            # DF meta
    df = pd.concat([M, X, y], axis=1)                 # Fusion finale

    print(f"Dataset : {df.shape[0]} lignes, {len(FEATURE_ORDER)} features. Répartition classes:")
    print(df["label"].value_counts().to_string())     # Distribution de y
    return df

# ==============================
# Modèle
# ==============================
def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),            # Normalise les features
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))    # Régression logistique équilibrée
    ])

def train_and_save(events_csv: str, model_out: str = DEFAULT_MODEL_PATH):
    df = build_training_frame(events_csv)             # Construit le dataset
    if df["label"].nunique() < 2:                     # Vérifie pluralité des classes
        print("\n❌ Toujours une seule classe après tentative de génération de négatifs.")
        print("   -> Vérifie ton CSV ou augmente NEG_PER_POS.")
        return

    X = df[FEATURE_ORDER].fillna(0.0).values          # Matrice de features (NA→0)
    y = df["label"].values                            # Labels

    pipe = make_pipeline()                             # Crée le pipeline
    pipe.fit(X, y)                                     # Entraîne le modèle

    # rapport “apparent” (sur train)
    preds = pipe.predict(X)                            # Prédictions sur train
    print("\n=== Rapport sur le set d'entraînement (indicatif) ===")
    print(classification_report(y, preds, digits=3))   # Rapport de perf

    joblib.dump({"model": pipe, "features": FEATURE_ORDER}, model_out)  # Sauvegarde pack
    print(f"\n✅ Modèle sauvegardé -> {model_out}")

def explain_model(model_path: str = DEFAULT_MODEL_PATH, top: int = 15):
    pack = joblib.load(model_path)                     # Charge le pack
    pipe: Pipeline = pack["model"]                     # Récup pipeline
    feats: List[str] = pack["features"]                # Liste de features
    clf: LogisticRegression = pipe.named_steps["clf"]  # Modèle logistique
    coefs = clf.coef_.ravel()                          # Coefficients
    idx = np.argsort(np.abs(coefs))[::-1]              # Tri par |coef| décroissant
    print("\nFeatures les plus influentes (|coef| décroissant):")
    for k in idx[:top]:
        print(f"{feats[k]:<18s}  coef={coefs[k]:+ .5f}")  # Affiche top-k

# ==============================
# Prédiction “maintenant”
# ==============================
def predict_now(model_path: str = DEFAULT_MODEL_PATH,
                symbols: Optional[List[str]] = None,
                out_csv: str = "pred_now.csv",
                min_prob: float = 0.5):
    pack = joblib.load(model_path)                     # Charge le pack
    pipe: Pipeline = pack["model"]                     # Récup pipeline
    feats: List[str] = pack["features"]                # Liste de features

    if symbols is None:
        if os.path.exists(DEFAULT_SYMBOLS_TXT):        # Lit symbols.txt si dispo
            with open(DEFAULT_SYMBOLS_TXT, "r", encoding="utf-8") as f:
                symbols = [ln.strip().upper() for ln in f if ln.strip()]
        else:
            symbols = ["AVAX","ARPA","NEBL","BRD","PIVX","ALGO","CHZ","FXS","EZ","NAS"]  # Liste par défaut

    ex = ccxt.binance({"enableRateLimit": True})       # Client Binance
    rows = []                                          # Vecteurs de features
    meta = []                                          # Meta (symbol)

    print(f"Prédiction live sur : {', '.join(symbols)}")
    for base in symbols:                               # Parcourt les tickers
        feats_now = features_for_symbol_now(ex, base)  # Features actuelles
        time.sleep(PAUSE_S)                            # Respecte rate limit
        if feats_now is None:
            print(f"  - {base:<8s} marché introuvable (USDT/BUSD). Ignoré.")  # Log marché absent
            continue
        rows.append([feats_now.get(k,0.0) for k in FEATURE_ORDER])  # Ajoute features
        meta.append({"symbol": base})                  # Ajoute meta

    if not rows:
        print("Aucun symbole exploitable.")            # Rien à scorer
        return

    X = pd.DataFrame(rows, columns=FEATURE_ORDER).fillna(0.0).values  # Matrice X
    prob = pipe.predict_proba(X)[:, 1]               # Proba de la classe “risque”
    out = pd.DataFrame(meta)                         # DF sorties
    out["prob_risk"] = prob                          # Ajoute proba
    out = out.sort_values("prob_risk", ascending=False)  # Trie décroissant

    print("\nTop (par proba pump-risk) :")
    for _, r in out.iterrows():                      # Affiche le ranking
        print(f" • {r['symbol']:<6s}  p={r['prob_risk']*100:5.1f}%")

    out.to_csv(out_csv, index=False)                 # Sauvegarde CSV
    print(f"\n📄 Résultats sauvegardés -> {out_csv}")
    if min_prob is not None:
        candidates = out[out["prob_risk"] >= float(min_prob)]  # Filtre sur seuil
        if not candidates.empty:
            print(f"\n⚠️  Candidats ≥ {min_prob:.2f} : " + ", ".join(candidates["symbol"].tolist()))  # Liste les alertes

# ==============================
# Menu sans sous-commande
# ==============================
def main_menu():
    choice = input("\nTon choix [1-4] (défaut 2): ").strip() or "2"  # Choix utilisateur
    if choice == "1":
        path = input(f"Chemin CSV événements [{DEFAULT_EVENTS_CSV}]: ").strip() or DEFAULT_EVENTS_CSV  # Chemin CSV
        out  = input(f"Chemin modèle sortie [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH   # Chemin modèle out
        train_and_save(path, out)                     # Entraîne et sauvegarde
    elif choice == "2":
        model = input(f"Chemin du modèle [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH  # Chemin modèle
        if not os.path.exists(model):
            print("\nModèle introuvable — entraînement rapide avec CSV par défaut…")
            train_and_save(DEFAULT_EVENTS_CSV, model) # Entraîne si besoin
        syms = input(f"Symboles (séparés par des virgules) [vide = {DEFAULT_SYMBOLS_TXT} ou liste par défaut]: ").strip()  # Liste de tickers
        symbols = [s.strip().upper() for s in syms.split(",") if s.strip()] if syms else None  # Parse liste
        thr = input("Seuil min de proba (0..1) [0.5]: ").strip()  # Seuil d’alerte
        min_prob = float(thr) if ok(thr) else 0.5    # Conversion seuil
        predict_now(model, symbols, "pred_now.csv", min_prob)  # Lance la prédiction
    elif choice == "3":
        model = input(f"Chemin du modèle [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH  # Chemin modèle
        top = input("Top N features [15]: ").strip()  # Nombre de features à afficher
        explain_model(model, int(top) if ok(top) else 15)  # Affiche l’importance des features
    else:
        print("Bye.")                                 # Sortie

if __name__ == "__main__":
    try:
        main_menu()                                   # Lance le menu principal
    except KeyboardInterrupt:
        print("\nInterrompu.")                        # Gestion Ctrl+C propre
