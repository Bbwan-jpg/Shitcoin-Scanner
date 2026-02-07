# app_streamlit.py
# UI Streamlit pour Solana Shitcoin Watcher — Stats + Bougies + Twitter + Risk Model + DB  # -> description générale
# Dépendances : streamlit, requests, pandas, plotly, python-dotenv, joblib, scikit-learn, sqlalchemy, passlib  # -> libs requises
# .env attendu :
#   DATABASE_URL=sqlite:///app.db               # -> URL de base de données
#   BIRDEYE_API_KEY=<optionnel>                 # -> clé API Birdeye (facultative)

import os, sys, time, pathlib, requests, datetime as dt  # Imports standards + HTTP + date/heure  # noqa: E401
import pandas as pd  # Manipulation de données tabulaires
import streamlit as st  # Framework UI
import plotly.graph_objects as go  # Graphiques Plotly (bougies)
from urllib.parse import urlparse  # Parsing d’URL
from dotenv import load_dotenv  # Chargement de .env
import json  # Sérialisation JSON
import re  # Expressions régulières

# --- rendre l'import local robuste
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # Ajoute le dossier courant au PYTHONPATH pour imports locaux

# --- charger .env
load_dotenv()  # Charge les variables d'environnement depuis .env
BIRDEYE_KEY = os.getenv("BIRDEYE_API_KEY")  # Récupère la clé Birdeye (ou None)

# --- moteur "stats"
from Solana import (  # Importe les fonctions de collecte/score depuis un module local
    fetch_pairs,        # récupère les pairs via DexScreener search
    get_metrics,        # extrait toutes les métriques utiles (inclut 'mint')
    calculate_score,    # calcule le score (0..100)
    is_safe             # filtre de sécurité basique
)

# --- modèle de risque
from risk_model import load_model, predict_pairs, explain_one  # Fonctions pour charger modèle, prédire, expliquer

# --- database
from db import (  # Fonctions d’accès à la BD
    init_db, create_user, authenticate,
    save_preset, list_presets,
    add_watchlist, list_watchlist,
    save_scan_rows, list_history, log_action
)

# ========== Endpoints externes ==========
DEX_PAIR_BY_ID = "https://api.dexscreener.com/latest/dex/pairs/solana/{pairAddress}"  # URL DexScreener (pair)
DEX_TOKENS     = "https://api.dexscreener.com/tokens/v1/{chainId}/{tokenAddresses}"  # URL DexScreener (tokens batch)

BIRDEYE_OHLCV_PAIR  = "https://public-api.birdeye.so/defi/v3/ohlcv/pair"  # Endpoint Birdeye OHLCV par pair
BIRDEYE_OHLCV_TOKEN = "https://public-api.birdeye.so/defi/v3/ohlcv"       # Endpoint Birdeye OHLCV par token

# ========== Helpers HTTP/Charts ==========
def get_pair_by_id(pair_address: str) -> dict | None:  # Récupère les infos de pair DexScreener
    """Récupère le JSON DexScreener d'un pair (pour prix courant & base mint)."""
    try:
        r = requests.get(DEX_PAIR_BY_ID.format(pairAddress=pair_address), timeout=8)  # GET avec timeout
        r.raise_for_status()  # Lève exception si code != 200
        j = r.json()  # Parse JSON
        pairs = j.get("pairs") if isinstance(j, dict) else None  # Extrait la liste "pairs"
        if isinstance(pairs, list) and pairs:
            return pairs[0]  # Renvoie le premier élément s'il existe
    except Exception:
        pass  # Silencieusement ignore les erreurs
    return None  # None si échec

def _birdeye_headers():  # Construit l’entête HTTP pour Birdeye
    return {"X-API-KEY": BIRDEYE_KEY, "x-chain": "solana", "accept": "application/json"}  # Headers API

def _to_candle_df(payload: dict) -> pd.DataFrame | None:  # Convertit la réponse Birdeye en DataFrame OHLCV
    d = payload.get("data", payload) if isinstance(payload, dict) else {}  # Supporte différentes enveloppes
    items = d.get("items", d.get("candles", d if isinstance(d, list) else []))  # Cherche la liste de bougies
    if not isinstance(items, list) or not items:
        return None  # Pas de données utilisables
    rows = []  # Accumule les lignes
    for it in items:  # Pour chaque bougie
        ts = it.get("unixTime") or it.get("time") or it.get("t") or it.get("startTime")  # Timestamp possible
        o  = it.get("o") or it.get("open")   # Open
        h  = it.get("h") or it.get("high")   # High
        l  = it.get("l") or it.get("low")    # Low
        c  = it.get("c") or it.get("close")  # Close
        v  = it.get("v") or it.get("volume") # Volume (peut être None)
        if ts is None or o is None or h is None or l is None or c is None:
            continue  # Ignore si info essentielle manquante
        rows.append([pd.to_datetime(int(ts), unit="s", utc=True), float(o), float(h), float(l), float(c), float(v or 0)])  # Ajoute ligne
    if not rows: return None  # DataFrame vide si rien
    df = pd.DataFrame(rows, columns=["t","open","high","low","close","volume"]).set_index("t")  # Crée DF indexée par temps
    return df.sort_index()  # Trie par index temporel

def fetch_candles_birdeye(pair_address: str, base_mint: str | None, timeframe: str, lookback_minutes: int) -> pd.DataFrame | None:
    """Essaie d'abord OHLCV par pair (pool), puis par token (mint)."""
    if not BIRDEYE_KEY:
        return None  # Pas de clé => pas de requête
    now = int(time.time()); start = now - lookback_minutes * 60  # Fenêtre temporelle (en secondes)
    try:
        r = requests.get(BIRDEYE_OHLCV_PAIR, headers=_birdeye_headers(),
                         params={"address": pair_address, "timeframe": timeframe, "from": start, "to": now}, timeout=10)  # Requête pair
        if r.ok:
            df = _to_candle_df(r.json())  # Conversion en DataFrame
            if df is not None: return df  # Retourne si OK
    except Exception:
        pass  # Ignore erreur pair
    if base_mint:
        try:
            r = requests.get(BIRDEYE_OHLCV_TOKEN, headers=_birdeye_headers(),
                             params={"address": base_mint, "timeframe": timeframe, "from": start, "to": now}, timeout=10)  # Requête token
            if r.ok: return _to_candle_df(r.json())  # Retourne DF si possible
        except Exception:
            pass  # Ignore erreur token
    return None  # Rien trouvé

# ========== Twitter helpers (via DexScreener /tokens/v1) ==========
def _batch(lst, n=30):  # Générateur pour batcher une liste par paquets de n
    for i in range(0, len(lst), n):
        yield lst[i:i+n]  # Sous-liste de taille n

def normalize_twitter(handle_or_url: str) -> str:  # Normalise un handle ou une URL X/Twitter en "@handle"
    s = (handle_or_url or "").strip()  # Nettoie l'entrée
    if not s: return ""  # Vide -> ""
    if s.startswith("http://") or s.startswith("https://"):  # C'est une URL ?
        try:
            u = urlparse(s)  # Parse l'URL
            host = (u.netloc or "").lower()  # Domaine
            path = (u.path or "").strip("/")  # Chemin
            if "twitter.com" in host or "x.com" in host:  # Si domaine Twitter/X
                handle = path.split("/")[0]  # Prend le premier segment
                return "@" + handle.lstrip("@")  # Retourne sous forme @handle
            return s  # Sinon renvoie tel quel
        except Exception:
            return s  # En cas d'erreur, renvoie tel quel
    return "@" + s.lstrip("@")  # Si simple handle, assure le préfixe @

def dexs_token_twitter_map(addresses: list[str]) -> dict[str, str]:  # Mappe mint -> handle Twitter via DexScreener
    """Retourne {mint -> @twitter} via /tokens/v1/solana/{addr1,addr2,...} (batch 30)."""
    out: dict[str, str] = {}  # Dictionnaire résultat
    addrs = [a for a in addresses if a]  # Filtre les vides
    for chunk in _batch(addrs, 30):  # Batches de 30
        url = DEX_TOKENS.format(chainId="solana", tokenAddresses=",".join(chunk))  # Construit l'URL
        try:
            r = requests.get(url, timeout=12)  # Appel HTTP
            r.raise_for_status()  # Lève si erreur
            payload = r.json()  # JSON
        except Exception:
            continue  # Passe au batch suivant en cas d'erreur
        items = payload.get("pairs") if isinstance(payload, dict) and "pairs" in payload else (payload if isinstance(payload, list) else [])  # Normalise
        if not isinstance(items, list):
            continue  # Ignore si format inattendu
        for it in items:  # Parcourt les résultats
            base = it.get("baseToken") or {}  # Bloc base token
            mint = base.get("address")  # Adresse mint
            info = it.get("info", {}) or {}  # Bloc info
            socials = info.get("socials", []) or []  # Réseaux sociaux
            tw = ""  # Valeur par défaut
            for s in socials:  # Parcourt les liens sociaux
                plat = (s.get("platform") or "").lower()  # Plateforme
                h = s.get("handle") or s.get("url") or ""  # Handle ou URL
                if plat in ("twitter", "x") or "twitter.com" in h or "x.com" in h:  # S'il s'agit de Twitter/X
                    tw = normalize_twitter(h); break  # Normalise et stop
            if mint and tw and mint not in out:
                out[mint] = tw  # Enregistre la correspondance
        time.sleep(0.05)  # Petite pause anti-rate limit
    return out  # Renvoie le mapping

def twitter_url(handle: str) -> str:  # Construit l’URL publique X à partir d’un @handle
    if not handle: return ""  # Vide -> ""
    h = handle.lstrip("@")  # Retire '@'
    return f"https://x.com/{h}"  # Construit l'URL

# Liste de chemins X à ignorer (pas des handles)
INVALID_X_PATHS = {"i", "home", "intent", "share", "explore", "settings", "notifications"}

def is_valid_twitter_handle(handle_or_url: str) -> bool:  # Valide qu'un handle/URL correspond à un vrai pseudo
    s = (handle_or_url or "").strip()  # Nettoie
    if not s:
        return False  # Vide => invalide
    # If it's a URL, normalize first
    if s.startswith("http://") or s.startswith("https://"):
        s = normalize_twitter(s)  # Normalise l'URL en @handle
    h = s.lstrip("@")  # Retire '@'
    if not h:
        return False  # Vide
    if h.lower() in INVALID_X_PATHS:
        return False  # Exclut chemins réservés
    # Twitter handle rules: letters, numbers, underscore, up to 15 chars
    return re.fullmatch(r"[A-Za-z0-9_]{1,15}", h) is not None  # Vérifie le pattern

def link_badge(url: str, label: str, outline: bool = True):  # Rend un badge lien propre en HTML/CSS
    """Petit bouton/lien propre (badge)."""
    if not url:
        return  # Rien si URL vide
    # Insertion d'un lien stylé (safe_allow_html=True autorise HTML)
    st.markdown(f"""
    <a href="{url}" target="_blank" rel="noopener noreferrer"
       style="
         display:inline-flex;align-items:center;gap:.5rem;
         padding:.35rem .7rem;border:{'1px solid #E2E8F0' if outline else '0'};
         border-radius:999px;background:#fff;text-decoration:none;
         font-weight:600;color:#111;box-shadow:0 1px 2px rgba(0,0,0,.06);
       ">
      {label}
    </a>
    """, unsafe_allow_html=True)

def twitter_badge(url: str, handle: str):  # Rend un badge X cliquable avec icône
    if not url:
        return  # Rien si URL vide
    h = handle.lstrip("@")  # Nettoie handle
    # Lien avec logo X (SVG inline)
    st.markdown(f"""
    <a href="{url}" target="_blank" rel="noopener"
       style="
         display:inline-flex;align-items:center;gap:.5rem;
         padding:.35rem .7rem;border:1px solid #E2E8F0;border-radius:999px;
         background:#fff;text-decoration:none;font-weight:600;color:#111;
         box-shadow:0 1px 2px rgba(0,0,0,.06);
       ">
      <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M18.244 2H21l-6.52 7.46L22 22h-6.844l-4.77-6.176L4.8 22H2l7.02-8.02L2 2h6.844l4.33 5.6L18.244 2Zm-2.4 18h2.16L8.22 4H6.06l9.784 16Z" fill="currentColor"/>
      </svg>
      @{h}
    </a>
    """, unsafe_allow_html=True)

def twitter_badge_crossed():  # Rend un badge X non cliquable avec pastille rouge (croix)
    # Bouton X non cliquable avec une petite pastille rouge intégrée
    st.markdown("""
    <span style="
      display:inline-flex;align-items:center;gap:.5rem;
      padding:.35rem .7rem;border:1px solid #E2E8F0;border-radius:999px;
      background:#fff;font-weight:600;color:#111;
      box-shadow:0 1px 2px rgba(0,0,0,.06);
    ">
      <!-- Logo X -->
      <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M18.244 2H21l-6.52 7.46L22 22h-6.844l-4.77-6.176L4.8 22H2l7.02-8.02L2 2h6.844l4.33 5.6L18.244 2Zm-2.4 18h2.16L8.22 4H6.06l9.784 16Z" fill="currentColor"/>
      </svg>
      <!-- Pastille rouge (croix) -->
      <span style="
        display:inline-flex;align-items:center;justify-content:center;
        width:18px;height:18px;border-radius:999px;background:#DC2626;
        box-shadow:0 1px 2px rgba(0,0,0,.1);
      ">
        <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M18 6L6 18M6 6l12 12" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </span>
    </span>
    """, unsafe_allow_html=True)

# ========== UI ==========
def main():  # Point d'entrée Streamlit
    st.set_page_config(page_title="Solana Shitcoin Watcher — Stats", layout="wide")  # Configure le layout/page

    # Init DB
    init_db()  # Crée les tables si absentes

    # --- preset en attente avant de créer les widgets
    if "_pending_preset" in st.session_state:  # Si un preset a été chargé et stocké temporairement
        for k, v in st.session_state["_pending_preset"].items():  # Réinjecte chaque clé/valeur dans la session
            st.session_state[k] = v  # Affectation
        del st.session_state["_pending_preset"]  # Nettoie la clé temporaire

    # --- Auth UI ---
    if "user" not in st.session_state:  # Initialise la clé user en session si absente
        st.session_state.user = None  # Valeur par défaut : non connecté

    if st.session_state.user is None:  # Si pas connecté, affiche l'écran d’auth
        # Création de 3 colonnes pour centrer la colonne du milieu
        # Le ratio [1, 2, 1] permet d'avoir un cadre qui prend 50% de la largeur
        left_co, cent_co, last_co = st.columns([1, 2, 1])  # Mise en page centrée
        
        with cent_co:  # Colonne centrale
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)  # Conteneur HTML
            st.markdown('<h1 class="auth-header"> SOLANA WATCHER</h1>', unsafe_allow_html=True)  # Titre
            
            tab_login, tab_signup = st.tabs(["Connexion", "Créer un compte"])  # Deux onglets : login/signup
            
            with tab_login:  # Onglet Connexion
                u = st.text_input("Username", key="login_user")  # Saisie username
                p = st.text_input("Mot de passe", type="password", key="login_pass")  # Saisie password
                if st.button("Se connecter", key="btn_login"):  # Bouton login
                    user = authenticate(u, p)  # Vérifie les identifiants
                    if user:
                        st.session_state.user = {"id": user.id, "username": user.username}  # Stocke l'utilisateur (id, username)
                        log_action(user.id, "login")  # Log l'action
                        st.rerun()  # Recharge l'app (post-login)
                    else:
                        st.error("Identifiants invalides.")  # Message d'erreur
            
            with tab_signup:  # Onglet Création
                u_new = st.text_input("Nouvel utilisateur", key="reg_user")  # Saisie nouveau username
                p_new = st.text_input("Nouveau mot de passe", type="password", key="reg_pass")  # Saisie nouveau password
                if st.button("Créer le compte", key="btn_reg"):  # Bouton création
                    if u_new and p_new:
                        ok, msg = create_user(u_new, p_new)  # Tente de créer l'utilisateur
                        if ok:
                            st.success("Compte créé ! Connecte-toi.")  # Succès
                        else:
                            st.error(msg)  # Erreur côté création
                    else:
                        st.warning("Remplis tous les champs.")  # Alerte champs vides
            
            st.markdown('</div>', unsafe_allow_html=True)  # Ferme le conteneur HTML
        
        # On arrête l'exécution ici si l'utilisateur n'est pas connecté
        st.stop()  # Stop l'app pour ne pas afficher le reste

    # --- Fin Auth UI ---

    # Bouton de déconnexion dans la sidebar une fois connecté
    st.sidebar.markdown(f"👤 **{st.session_state.user['username']}**")  # Affiche le nom d'utilisateur dans la sidebar
    if st.sidebar.button("Se déconnecter"):  # Bouton logout
        log_action(st.session_state.user["id"], "logout")  # Log l'action
        st.session_state.user = None  # Supprime l’utilisateur de la session
        st.rerun()  # Recharge l’app pour revenir à l’écran login

    # --- Sidebar ---
    st.sidebar.title("⚙️ Paramètres (Stats)")  # Titre de la sidebar
    with st.sidebar.expander("Source & clés", expanded=True):  # Panneau sources/clé
        st.write("• Source : DexScreener (search)")  # Info source
        st.write(f"• Birdeye pour bougies Plotly : {'✅ détectée' if BIRDEYE_KEY else '❌ absente'}")  # Statut clé Birdeye

    # ——— Risque (modèle supervisé)
    st.sidebar.markdown("### 🤖 Modèle de risque")  # Section modèle
    use_model = st.sidebar.toggle("Activer le modèle", value=True, key="use_model")  # Toggle d’activation modèle
    model_path = st.sidebar.text_input("Chemin modèle", value="model_pump.pkl", key="model_path", help="Fichier .pkl exporté par pump_learn.py")  # Saisie chemin

    # --- Filtres principaux
    min_score = st.sidebar.number_input("Score minimum", min_value=0.0, value=30.0, step=1.0, key="min_score")  # Seuil score
    apply_safety = st.sidebar.toggle("Filtre sécurité is_safe()", value=True, key="apply_safety",
                                    help="Anti-rug basique (liquidité trop basse/haute, 0 ventes, activité faible)")  # Toggle sécurité
    no_age_filter = st.sidebar.toggle("Ignorer le filtre d'âge", value=True, key="no_age_filter")  # Toggle âge
    age_max = st.sidebar.number_input("Âge max (minutes)", min_value=0, value=60, step=5, key="age_max")  # Seuil âge
    liq_min = st.sidebar.number_input("Liquidité min ($)", min_value=0, value=500, step=100, key="liq_min")  # Seuil liq min
    liq_max = st.sidebar.number_input("Liquidité max ($)", min_value=0, value=500_000, step=5_000, key="liq_max")  # Seuil liq max
    min_buy_pressure = st.sidebar.slider("Pression d'achat min (%)", 0, 100, 0, step=5, key="min_buy_pressure")  # Seuil pression achat
    min_vol_liq = st.sidebar.slider("Vol/Liq 5m min (%)", 0, 100, 0, step=1, key="min_vol_liq")  # Seuil vol/liq
    min_m5 = st.sidebar.slider("Pump 5m min (%)", 0, 200, 0, step=1, key="min_m5")  # Seuil m5
    topn = st.sidebar.slider("Top candidats affichés", 5, 100, 20, step=1, key="topn")  # Nombre de lignes affichées

    # --- Presets (sauvegarder/charger paramètres)
    st.sidebar.markdown("### 💾 Presets")  # Section presets
    preset_name = st.sidebar.text_input("Nom du preset")  # Saisie nom preset
    col_ps1, col_ps2 = st.sidebar.columns(2)  # Deux colonnes pour actions presets

    with col_ps1:  # Colonne gauche : enregistrer
        if st.button("Enregistrer preset"):  # Bouton save preset
            params = {  # Prépare dict des paramètres à sauvegarder
                "use_model": st.session_state.use_model,
                "model_path": st.session_state.model_path,
                "min_score": st.session_state.min_score,
                "apply_safety": st.session_state.apply_safety,
                "no_age_filter": st.session_state.no_age_filter,
                "age_max": st.session_state.age_max,
                "liq_min": st.session_state.liq_min,
                "liq_max": st.session_state.liq_max,
                "min_buy_pressure": st.session_state.min_buy_pressure,
                "min_vol_liq": st.session_state.min_vol_liq,
                "min_m5": st.session_state.min_m5,
                "topn": st.session_state.topn,
            }
            ok, msg = save_preset(st.session_state.user["id"], preset_name or "preset", params)  # Sauvegarde en DB
            _ = st.success(msg) if ok else st.warning(msg)  # Feedback UI

    with col_ps2:  # Colonne droite : charger
        presets = list_presets(st.session_state.user["id"])  # -> objets ORM  # Lit les presets existants
        if presets:
            preset_labels = [f"{p.name} — {p.updated_at:%Y-%m-%d %H:%M}" for p in presets]  # Labels lisibles
            idx_preset = st.selectbox(  # Select d’un preset
                "Charger preset",
                list(range(len(presets))),
                format_func=lambda i: preset_labels[i]
            )
            if st.button("Charger"):  # Bouton charger
                params = json.loads(presets[idx_preset].params_js)  # Parse JSON des paramètres
                # On pousse dans _pending_preset et on relance l'app
                st.session_state["_pending_preset"] = params  # Stock temporaire
                st.rerun()  # Relance pour appliquer
        else:
            st.caption("Aucun preset enregistré.")  # Message si aucun

    refresh = st.sidebar.button("🔄 Scanner maintenant")  # Bouton pour déclencher un scan

    # --- Header ---
    st.title("Solana Shitcoin Watcher")  # Titre principal

    # --- Charger le modèle (si activé) ---
    pack = None  # Par défaut pas de modèle
    if st.session_state.use_model:  # Si toggle actif
        try:
            pack = load_model(st.session_state.model_path)  # Charge le .pkl
            st.sidebar.success("Modèle chargé.")  # OK
        except Exception as e:
            st.sidebar.warning(f"Impossible de charger {st.session_state.model_path} : {e}")  # Alerte
            pack = None  # Désactive modèle si échec

    # --- Scan ---
    if refresh or "rows_raw" not in st.session_state:  # Si demande refresh ou première fois
        with st.spinner("Scan en cours…"):  # Spinner UI
            # 1) pairs bruts pour Risk% + future explication
            pairs = fetch_pairs()  # Récupère les pairs (DexScreener)
            st.session_state.pairs_raw = pairs  # Stocke bruts pour plus tard

            # 2) Risk predictions (si modèle chargé)
            risk_map = {}  # Dictionnaire pair -> prob_risk
            if pack and pairs:
                risk_df = predict_pairs(pairs, pack)  # columns: pair,symbol,url,prob_risk  # Prédiction modèle
                risk_map = {row["pair"]: float(row["prob_risk"]) for _, row in risk_df.iterrows()}  # Map de probabilité
            st.session_state.risk_map = risk_map  # Stocke en session

            # 3) pipeline stats habituel
            rows = []  # Liste des métriques retenues
            for p in pairs:  # Parcourt chaque pair brut
                m = get_metrics(p)  # Calcule métriques
                if not st.session_state.no_age_filter and m.get("age", 0) > st.session_state.age_max:
                    continue  # Filtre âge si activé
                if not (st.session_state.liq_min <= m["liq"] <= st.session_state.liq_max):
                    continue  # Filtre liq
                if m["buy_pressure"] < st.session_state.min_buy_pressure:
                    continue  # Filtre pression achat
                if m["vol_liq_ratio"] < st.session_state.min_vol_liq:
                    continue  # Filtre vol/liq
                if m["m5"] < st.session_state.min_m5:
                    continue  # Filtre pump 5m
                if st.session_state.apply_safety and not is_safe(m):
                    continue  # Filtre sécurité
                sc = calculate_score(m)  # Score custom
                if sc < st.session_state.min_score:
                    continue  # Filtre score min
                m["_score"] = round(sc, 1)  # Arrondit le score
                rows.append(m)  # Conserve la ligne

            rows.sort(key=lambda x: x["_score"], reverse=True)  # Trie par score décroissant
            st.session_state.rows_raw = rows  # Stocke la sélection

            # 4) Enregistrer l'historique du scan pour l'utilisateur
            try:
                n_saved, _ = save_scan_rows(st.session_state.user["id"], rows, st.session_state.risk_map)  # Persiste historique
                if n_saved:
                    log_action(st.session_state.user["id"], "scan_saved", {"count": n_saved})  # Log l'action
            except Exception as e:
                st.sidebar.warning(f"Historique non enregistré : {e}")  # Alerte si échec DB

    rows = st.session_state.get("rows_raw", [])  # Récupère lignes retenues
    risk_map = st.session_state.get("risk_map", {})  # Récupère map risque
    pairs_raw = st.session_state.get("pairs_raw", [])  # Récupère pairs bruts

    if not rows:  # Si aucune ligne
        st.info("Aucun candidat avec les seuils actuels. Dessers un peu les filtres (score/liq/press/pump).")  # Message info
        st.stop()  # Stop UI

    # --- Twitter enrichment (pour la vue liste) ---
    mints_for_topn = [m.get("mint") for m in rows[:st.session_state.topn] if m.get("mint")]  # Liste des mints pour top N
    tw_map = dexs_token_twitter_map(mints_for_topn) if mints_for_topn else {}  # Map mint -> @twitter

    # --- Tableau ---
    def to_row(m: dict) -> dict:  # Transforme une ligne métrique en dict pour DataFrame
        handle = tw_map.get(m.get("mint",""), "")  # Récupère handle depuis map
        if not is_valid_twitter_handle(handle):
            handle = ""  # Invalide -> vide
        link   = twitter_url(handle) if handle else ""  # Construit l'URL Twitter si handle
        risk = risk_map.get(m["address"])  # Récupère prob_risk par pair
        return {
            "Symbole": m["symbol"],
            "Score": m["_score"],
            "Risk %": (round(risk*100, 1) if isinstance(risk, float) else None),
            "m5 %": round(m["m5"], 2),
            "Pression achat %": round(m["buy_pressure"], 1),
            "Vol/Liq % (5m)": round(m["vol_liq_ratio"], 1),
            "Liq $": int(m["liq"]),
            "Tx 5m": int(m["tx_total"]),
            "Buys": int(m["buys"]),
            "Sells": int(m["sells"]),
            "Âge (min)": round(m["age"], 1),
            "FDV": int(m["fdv"]) if isinstance(m["fdv"], (int, float)) else m["fdv"],
            "Pair": m["address"],
            "Lien": m.get("url") or f"https://dexscreener.com/solana/{m['address']}",
            "Twitter": handle,
            "Lien X": link
        }

    rows_view = [to_row(m) for m in rows[:st.session_state.topn]]  # Construit la vue sur top N
    st.subheader("Candidats")  # Titre section
    df_view = pd.DataFrame(rows_view)  # DataFrame pour affichage
    st.dataframe(  # Affiche le tableau interactif
        df_view,
        width="stretch",
        hide_index=True,
        column_config={
            "Lien": st.column_config.LinkColumn("Dexscreener", display_text="Ouvrir"),  # Colonne lien Dex
            "Lien X": st.column_config.LinkColumn("Twitter", display_text="Profil X"),   # Colonne lien X
            # optionnel: cacher la colonne brute si tu veux
            # "Pair": None,
        },
    )

    # --- Suivi live (prix spot) ---
    st.subheader("📈 Suivi live d’un pair (prix spot)")  # Titre section live
    forced = st.session_state.get("force_pair")  # Récupère un pair forcé depuis l'historique (si défini)
    options = [f"{r['Symbole']} | {r['Pair']}" for r in rows_view]  # Options lisibles pour select

    initial_index = 0  # Index par défaut
    if forced:
        for i, r in enumerate(rows_view):  # Parcourt pour trouver l'index du pair forcé
            if r["Pair"] == forced:
                initial_index = i  # Met l'index trouvé
                break
    # on consomme le “force_pair” après usage
    if "force_pair" in st.session_state:
        del st.session_state["force_pair"]  # Nettoie le flag après lecture

    idx = st.selectbox(  # Select du token à suivre
        "Choisis un token",
        list(range(len(options))),
        format_func=lambda i: options[i],
        index=initial_index,
    )

    chosen_m = rows[idx]  # même index car rows_view garde l'ordre  # Récupère la ligne choisie
    pair_address = chosen_m["address"]  # Adresse du pair
    pair_link = chosen_m.get("url") or f"https://dexscreener.com/solana/{pair_address}"  # Lien Dex
    raw_handle = tw_map.get(chosen_m.get("mint",""), "")  # Handle brut depuis map
    chosen_handle = raw_handle if is_valid_twitter_handle(raw_handle) else ""  # Valide le handle ou vide

    c1, c2, c3 = st.columns(3)  # 3 colonnes d’infos
    with c1:
        st.write(f"**{chosen_m['symbol']}**")  # Affiche le symbole
        link_badge(pair_link, "Ouvrir sur Dexscreener")  # Badge lien Dex
        if chosen_handle:
            twitter_badge(twitter_url(chosen_handle), chosen_handle)  # Badge X cliquable
        else:
            twitter_badge_crossed()  # Badge X barré (non trouvé)

    with c2:
        st.metric("Score", value=chosen_m["_score"], delta=f"{chosen_m['m5']:+.2f}% (5m)")  # Tuile score + delta m5
    with c3:
        risk_val = risk_map.get(pair_address)  # Probabilité de risque pour le pair
        if risk_val is not None:
            st.metric("Risk % (modèle)", value=f"{risk_val*100:.1f}%")  # Tuile risk %
        st.metric("Liq ($)", value=int(chosen_m["liq"]), delta=f"BuyPr {chosen_m['buy_pressure']:.1f}%")  # Tuile liq + pression

    # --- Ajout Watchlist ---
    st.markdown("#### ⭐ Watchlist")  # Sous-titre watchlist
    note_val = st.text_input("Note (optionnel)", value="")  # Champ note
    if st.button("Ajouter le token sélectionné à ma watchlist"):  # Bouton ajout
        try:
            ok, msg = add_watchlist(  # Appel DB
                user_id=st.session_state.user["id"],
                symbol=chosen_m.get("symbol", "?"),
                pair_addr=pair_address,
                mint=chosen_m.get("mint"),
                notes=note_val
            )
            if ok:
                st.success(msg)             # Feedback succès
            else :
                st.warning(msg)             # Feedback avertissement
            log_action(st.session_state.user["id"], "watch_add", {"pair": pair_address, "symbol": chosen_m.get("symbol")})  # Log
        except Exception as e:
            st.error(f"Erreur watchlist : {e}")  # Erreur DB

    # --- Affichage Watchlist utilisateur ---
    wl_objs = list_watchlist(st.session_state.user["id"])  # Lit la watchlist
    if wl_objs:
        st.markdown("#### 📌 Ma watchlist")  # Titre watchlist
        wl_rows = [{  # Transforme en dicts
            "id": w.id,
            "symbol": w.symbol,
            "pair": getattr(w, "pair_addr", ""),
            "mint": w.mint,
            "notes": w.notes,
            "created_at": w.created_at,
        } for w in wl_objs]
        st.dataframe(pd.DataFrame(wl_rows), width="stretch", hide_index=True)  # Affiche la liste

    st.markdown("---")  # Séparateur
    dur_min = st.slider("Durée du suivi (minutes)", 1, 30, 5)  # Durée live
    interval = st.slider("Intervalle (sec)", 2, 30, 5)  # Intervalle de rafraîchissement
    go = st.toggle("Démarrer le suivi live", value=False)  # Toggle démarrage live
    chart_ph = st.empty()  # Placeholder graphique
    price_ph = st.empty()  # Placeholder métrique prix

    if go and pair_address:  # Si live activé et pair défini
        key = f"ts_{pair_address}"  # Clé session pour la série temporelle
        if key not in st.session_state:
            st.session_state[key] = []  # Initialise la liste si absente
        end = time.time() + 60 * dur_min  # Calcule le timestamp de fin
        with st.spinner("Live…"):  # Spinner UI
            while time.time() < end:  # Boucle jusqu'à fin
                cur = get_pair_by_id(pair_address)  # Récupère le prix courant
                price = float((cur or {}).get("priceUsd") or 0.0)  # Extrait priceUsd
                # >>> timezone-aware UTC (remplace pd.Timestamp.utcnow())
                st.session_state[key].append({"t": pd.Timestamp.now(tz=dt.timezone.utc), "price": price})  # Ajoute point
                df_ts = pd.DataFrame(st.session_state[key]).set_index("t")  # Construit DF indexée temps
                chart_ph.line_chart(df_ts["price"])  # Affiche courbe temps réel
                price_ph.metric("Prix (USD)", value=round(price, 8))  # Affiche valeur actuelle
                time.sleep(interval)  # Pause entre mesures

    # --- Bougies ---
    st.markdown("---")  # Séparateur
    st.subheader("🕯️ Bougies du pair")  # Titre section bougies

    renderer = st.radio(  # Choix du moteur d’affichage des bougies
        "Rendu du graphe",
        ["Iframe (GeckoTerminal)", "Plotly (Birdeye)"],
        horizontal=True,
        help="Iframe = sans clé. Plotly = nécessite BIRDEYE_API_KEY."
    )

    pair_json = get_pair_by_id(pair_address) or {}  # Récupère à nouveau le pair complet
    base_mint = ((pair_json.get("baseToken") or {}).get("address")) if isinstance(pair_json, dict) else None  # Base mint si dispo

    if renderer.startswith("Iframe"):  # Choix GeckoTerminal
        gtc_url = f"https://www.geckoterminal.com/solana/pools/{pair_address}?embed=1&info=0&swaps=0"  # URL iframe
        st.components.v1.iframe(gtc_url, height=540)  # Intègre l’iframe
        st.caption("Chart intégré via GeckoTerminal (mode sans clé).")  # Légende
    else:  # Choix Plotly (Birdeye)
        if not BIRDEYE_KEY:
            st.info("Ajoute une clé Birdeye (`BIRDEYE_API_KEY`) pour activer les bougies Plotly. Sinon, utilise le mode Iframe.")  # Info manque de clé
        else:
            colA, colB = st.columns(2)  # Deux colonnes de contrôles
            with colA:
                timeframe = st.selectbox("Intervalle", ["1m","5m","15m","1h","4h","1d"], index=0)  # Choix timeframe
            with colB:
                lookback = st.slider("Historique (minutes)", 30, 1440, 240, step=30)  # Choix historique
            with st.spinner("Récupération des bougies…"):  # Spinner
                df = fetch_candles_birdeye(pair_address, base_mint, timeframe, lookback)  # Charge OHLCV
            if df is None or df.empty:
                st.warning("Pas de bougies retournées par l’API pour ce pair/intervalle (essaie un autre intervalle ou augmente l’historique).")  # Alerte
            else:
                fig = go.Figure(  # Crée la figure Plotly
                    data=[go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"])]
                )
                fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=0, r=0, t=20, b=0))  # Mise en forme
                st.plotly_chart(fig, width="stretch")  # Affiche le graphe

    # --- Explication du modèle pour le token sélectionné ---
    if pack:  # Si un modèle est chargé
        st.markdown("---")  # Séparateur
        with st.expander("🔍 Explication du modèle (features les plus influentes)", expanded=False):  # Expander repliable
            # Pour éviter de calculer tant que l'utilisateur n'a pas cliqué
            if st.button("Afficher l'explication", key=f"btn_expl_{pair_address}"):  # Bouton pour déclencher le calcul
                # retrouver l'objet pair brut pour le pair sélectionné
                raw_selected = next((p for p in pairs_raw if p.get("pairAddress") == pair_address), None)  # Cherche le pair brut
                if raw_selected is None:
                    st.info("Impossible de retrouver les données brutes de ce pair.")  # Info si introuvable
                else:
                    try:
                        df_expl = explain_one(raw_selected, pack, top=8)   # cols: feature, contrib, value  # Explique le modèle
                        st.dataframe(df_expl, width="stretch", hide_index=True)  # Affiche le tableau d'explications
                        st.caption("Les contributions (contrib) correspondent à la part de chaque feature dans le logit pour ce pair (approx).")  # Légende
                    except Exception as e:
                        st.info(f"Impossible de calculer l’explication pour ce pair : {e}")  # Alerte si erreur

    # --- Historique des scans (utilisateur) ---
    st.markdown("---")  # Séparateur
    st.subheader("📜 Historique de mes scans (derniers)")  # Titre section historique
    try:
        hist = list_history(st.session_state.user["id"], limit=200)  # Récupère l'historique depuis la DB
        if hist:
            hist_df = pd.DataFrame(hist).copy()  # DataFrame à partir de l'historique

            # Harmonisation des noms de colonnes les plus probables
            rename_map = {  # Mapping éventuel pour aligner les noms
                "pair_addr": "pair",
                "pairAddress": "pair",
                "created_at": "ts",
                "timestamp": "ts",
                "symbol": "symbol",
                "score": "score",
                "risk": "risk",
            }
            for k, v in rename_map.items():  # Applique les renommages si besoin
                if k in hist_df.columns and v not in hist_df.columns:
                    hist_df.rename(columns={k: v}, inplace=True)

            cols_show = [c for c in ["ts", "symbol", "pair", "score", "risk"] if c in hist_df.columns]  # Colonnes à afficher
            if "ts" in cols_show:
                hist_df.sort_values("ts", ascending=False, inplace=True)  # Trie par date décroissante

            st.dataframe(  # Affiche l'historique condensé
                hist_df[cols_show],
                width="stretch",
                hide_index=True
            )

            # Sélecteur rapide dans l'historique
            uniq = hist_df.dropna(subset=["pair"]).drop_duplicates("pair")  # Uniques par pair
            if not uniq.empty:
                opts = [f"{row.get('symbol','?')} | {row['pair']}" for _, row in uniq.iterrows()]  # Options lisibles
                sel = st.selectbox("Ouvrir depuis l'historique", list(range(len(uniq))), format_func=lambda i: opts[i])  # Select
                if st.button("Afficher ce token"):  # Bouton pour forcer l'ouverture
                    chosen_pair = uniq.iloc[sel]["pair"]  # Pair choisi
                    st.session_state["force_pair"] = chosen_pair  # Pose le flag en session
                    st.rerun()  # Relance pour sélectionner ce pair
            else:
                st.caption("Historique présent mais aucune pair exploitable.")  # Info si vide
        else:
            st.caption("Aucun scan enregistré pour le moment.")  # Info si pas d’historique
    except Exception as e:
        st.info(f"Historique indisponible : {e}")  # Alerte si erreur DB

    st.caption("Tous droits réservés, reproduction et diffusion interdite sous peine de poursuites")  # Mention légale

if __name__ == "__main__":  # Point d’exécution directe
    main()  # Lance l’app Streamlit
