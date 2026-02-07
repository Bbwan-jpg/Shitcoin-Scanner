
#A mettre dans un .env 
#BIRDEYE_API_KEY=32c218876aff440b8665c3a805a9307c

# sol_watcher_stat.py
import os, time, csv, argparse                    # OS/temps/CSV/CLI
import datetime as dt                             # Dates/horaires
from typing import Dict, Any, List, Optional      # Types d'annotation
import requests                                   # Requêtes HTTP
from urllib.parse import urlparse                 

# -----------------------------
# Config
# -----------------------------
DEX_SEARCH = "https://api.dexscreener.com/latest/dex/search"   # Endpoint de recherche DexScreener
DEX_QUERIES = ["solana", "pump.fun solana", "raydium", "meteora"]  # Mots-clés à scanner
CSV_PATH = "candidats_stats.csv"                                 # Fichier de log CSV

session = requests.Session()                                     # Session HTTP réutilisable
session.headers.update({"User-Agent": "sol-watcher-stat/1.5", "Accept": "application/json"})  # Headers par défaut

# -----------------------------
# Utils & Fetch
# -----------------------------
def get_json(url: str, params: Optional[dict] = None) -> dict:
    try:
        r = session.get(url, params=params or {}, timeout=10)    # GET avec timeout
        return r.json() if r.status_code == 200 else {}          # Parse JSON si 200, sinon dict vide
    except:
        return {}                                                # En cas d’erreur réseau/JSON

def fetch_pairs() -> List[Dict[str, Any]]:
    pairs = []                                                   # Accumulateur
    # On scanne large
    for q in DEX_QUERIES:                                        # Pour chaque requête
        data = get_json(DEX_SEARCH, params={"q": q})             # Appel API de recherche
        pairs.extend(data.get("pairs", []) or [])                # Ajoute les pairs trouvés
        time.sleep(0.1)                                          # Petite pause (politesse / rate limit)
    
    # Dédoublonnage
    uniq = {}                                                    # map pairAddress -> payload
    for p in pairs:
        pid = p.get("pairAddress")                               # Identifiant unique du pool
        if pid: uniq[pid] = p                                    # Garde la dernière occurrence
    return list(uniq.values())                                   # Liste dédupliquée

# -----------------------------
# NOUVELLES ANALYSES STATISTIQUES
# -----------------------------

def get_metrics(p: Dict[str, Any]) -> dict:
    """Extrait et calcule toutes les stats importantes."""
    
    # 1. Données brutes
    price_change = float((p.get("priceChange") or {}).get("m5", 0))   # Variation prix 5 min (%)
    liquidity = float((p.get("liquidity") or {}).get("usd", 0))       # Liquidité USD
    vol_m5 = float((p.get("volume") or {}).get("m5", 0))              # Volume 5 min
    fdv = float(p.get("fdv", 0))                                      # Fully Diluted Valuation
    
    created_at = int(p.get("pairCreatedAt", 0) or 0)                  # Timestamp de création (ms)
    age_min = (int(time.time() * 1000) - created_at) / 60000 if created_at else 0  # Âge du pool (min)
    
    # 2. Analyse des Transactions (Buy Pressure)
    txns = (p.get("txns") or {}).get("m5", {})                        # Bloc transactions sur 5 min
    buys = int(txns.get("buys", 0))                                   # Nb d’achats
    sells = int(txns.get("sells", 0))                                 # Nb de ventes
    total_tx = buys + sells                                           # Total trades
    
    buy_pressure = (buys / total_tx * 100) if total_tx > 0 else 0     # % d’achats
    
    # 3. Ratio Viralité (Volume / Liquidity)
    # Si le volume 5min dépasse 5% de la liquidité totale, c'est très actif
    vol_liq_ratio = (vol_m5 / liquidity * 100) if liquidity > 0 else 0  # Activité relative
    base_mint = (p.get("baseToken") or {}).get("address")               # Mint du token de base

    return {
        "symbol": (p.get("baseToken") or {}).get("symbol", "?"),  # Ticker
        "address": p.get("pairAddress"),                          # Adresse du pool
        "m5": price_change,                                       # Δ prix 5m (%)
        "liq": liquidity,                                         # Liquidité $
        "vol_m5": vol_m5,                                         # Volume 5m
        "mint": base_mint,                                        # Mint
        "age": age_min,                                           # Âge (min)
        "buys": buys,                                             # Nb achats
        "sells": sells,                                           # Nb ventes
        "tx_total": total_tx,                                     # Nb total tx
        "buy_pressure": buy_pressure,                             # % achats
        "vol_liq_ratio": vol_liq_ratio,                           # (vol5m/liq)%
        "fdv": fdv,                                               # FDV
        "url": p.get("url")                                       # Lien DexScreener
    }

def calculate_score(m: dict) -> float:
    """
    Score sur 100 pts.
    Critères : 
    - Le prix monte (30%)
    - Pression d'achat > 60% (40%)
    - Grosse activité par rapport à la liquidité (30%)
    """
    score = 0
    
    # A. Prix (max 30 pts)
    # Si m5 entre 0 et 20%, on donne des points. Si > 50% (trop pumpé), on calme.
    if m["m5"] > 0:
        score += min(30, m["m5"] * 2)                               # 2 pts par % jusqu’à 15% (cap à 30)
    
    # B. Pression d'achat (max 40 pts)
    # On veut > 50%. Idéalement 70%.
    if m["buy_pressure"] > 50:
        score += (m["buy_pressure"] - 50)                           # Chaque % au-dessus de 50 ajoute 1 pt
    
    # C. Viralité (max 30 pts)
    # Si le volume 5m est > 2% de la liquidité, ça bouge bien
    if m["vol_liq_ratio"] > 1:
        score += min(30, m["vol_liq_ratio"] * 5)                    # 5 pts par % (cap 30)

    return score

def is_safe(m: dict) -> bool:
    """Filtres de sécurité (Anti-Rug basique)"""
    # 1. Honeypot suspect : Beaucoup d'achats mais 0 vente ?
    if m["buys"] > 10 and m["sells"] == 0:
        return False
    
    # 2. Liquidité trop faible (Scam facile) ou trop haute (Stablecoin/Vieux token)
    if m["liq"] < 500 or m["liq"] > 500000:
        return False
        
    # 3. Morte activité
    if m["tx_total"] < 5:
        return False
        
    return True                                                     # Passe les filtres

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()                                  # Parseur d’arguments CLI
    ap.add_argument("--loop", action="store_true", help="Scan en continu")  # Mode boucle
    ap.add_argument("--min-score", type=float, default=30.0, help="Score min pour afficher")  # Seuil d’affichage
    args = ap.parse_args()                                          # Parse les arguments

    print(f"Stats-Scanner démarré... (Score min: {args.min_score})")  # Log de départ
    
    # Header CSV
    if not os.path.exists(CSV_PATH):                                # Crée le CSV si absent
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["time", "symbol", "score", "m5%", "buy_pressure%", "vol/liq%", "liquidity", "url"])  # En-têtes

    def run_once():
        raw_pairs = fetch_pairs()                                    # Récupère les pools
        candidates = []                                              # Candidats retenus

        for p in raw_pairs:
            m = get_metrics(p)                                       # Calcule les métriques
            
            if not is_safe(m):                                       # Filtre sécurité
                continue
            
            final_score = calculate_score(m)                         # Score pondéré
            
            if final_score >= args.min_score:                        # Garde si au-dessus du seuil
                candidates.append((m, final_score))

        # Tri par score décroissant
        candidates.sort(key=lambda x: x[1], reverse=True)            # Classement des meilleurs

        if candidates:
            print(f"\n--- {dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')} ---")  # Horodatage UTC
            
        for m, sc in candidates[:10]: # Top 10                        # Affiche jusqu’à 10 résultats
            # Emojis indicateurs
            pressure_icon = "🟢" if m["buy_pressure"] > 60 else ("🔴" if m["buy_pressure"] < 40 else "⚪")  # Code couleur
            
            print(f"🔥 {m['symbol']:<6} | Score: {int(sc):>3} | M5: {m['m5']:+5.1f}% | "
                  f"Press: {m['buy_pressure']:4.1f}% {pressure_icon} | "
                  f"Vol/Liq: {m['vol_liq_ratio']:4.1f}% | Liq: ${int(m['liq'])}")       # Ligne lisible
            print(f" 👉 {m['url']}")                                                    # Lien DexScreener
            
            # Save
            with open(CSV_PATH, "a", newline="") as f:                                   # Append dans le CSV
                csv.writer(f).writerow([
                    dt.datetime.now(dt.timezone.utc).isoformat(),                        # Timestamp ISO UTC
                    m['symbol'],                                                         # Symbole
                    f"{sc:.1f}",                                                         # Score formaté
                    m['m5'],                                                             # Δ 5m
                    f"{m['buy_pressure']:.1f}",                                          # Buy pressure %
                    f"{m['vol_liq_ratio']:.1f}",                                         # Vol/Liq %
                    m['liq'],                                                            # Liquidité
                    m['url']                                                             # Lien
                ])

    if args.loop:
        try:
            while True:                                               # Boucle infinie en mode --loop
                run_once()                                            # Un passage
                time.sleep(10)                                        # Pause entre scans
        except KeyboardInterrupt:
            print("Arrêt.")                                           # Sortie propre au Ctrl+C
    else:
        run_once()                                                    # Un seul scan si pas de --loop

if __name__ == "__main__":
    main()                                                            # Point d’entrée
