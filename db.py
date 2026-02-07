# db.py
# Base SQL: users, presets, watchlist, scan_history, action_logs  # -> description du module
# Dépendances : sqlalchemy, passlib, python-dotenv                  # -> dépendances utilisées
# .env: DATABASE_URL=sqlite:///app.db                                # -> variable d'environnement attendue

from __future__ import annotations  # Active l'évaluation différée des annotations (utile pour les relations ORM)

import os  # Accès aux variables d'environnement et au système de fichiers
import json
import datetime as dt  # Gestion de dates/heures (alias dt)
from typing import Dict, Any, List, Optional, Tuple  
from sqlalchemy import (  # Primitives SQLAlchemy (engine, colonnes, types, contraintes)
    create_engine, Column, Integer, String, DateTime, Text, Float,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker  # Base ORM, relations, fabrique de sessions

from passlib.hash import pbkdf2_sha256  # Hachage et vérification de mots de passe
from dotenv import load_dotenv  # Chargement des variables depuis .env

# ---------------------------------------------------------------------
# Helpers time (UTC aware)
# ---------------------------------------------------------------------
def utcnow() -> dt.datetime:  # Déclare une fonction utilitaire de temps en UTC "aware"
    """UTC timezone-aware 'now' (remplace datetime.utcnow())."""
    return dt.datetime.now(dt.timezone.utc)  # Renvoie la date/heure actuelle avec fuseau UTC

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
load_dotenv()  # Charge les variables d'environnement depuis un fichier .env si présent
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")  # Lit la chaîne de connexion DB (par défaut SQLite local)
DB_ECHO = os.getenv("DB_ECHO", "0") in ("1", "true", "True")  # Active le log SQL si DB_ECHO vaut 1/true

engine = create_engine(  # Crée l'engine SQLAlchemy (point d'accès à la BD)
    DATABASE_URL,  # URL de la base (SQLite, Postgres, etc.)
    future=True,  # Utilise l'API 2.0 de SQLAlchemy
    echo=DB_ECHO,  # Affiche les requêtes SQL si activé
    pool_pre_ping=True,  # Vérifie la connexion avant chaque utilisation
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)  # Fabrique de sessions DB
Base = declarative_base()  # Base déclarative pour définir les modèles ORM

# ---------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------
class User(Base):  # Définition du modèle/table "users"
    __tablename__ = "users"  # Nom de la table en base

    id         = Column(Integer, primary_key=True)  # Clé primaire
    username   = Column(String(64), unique=True, nullable=False)  # Identifiant unique
    password_h = Column(String(256), nullable=False)  # Hash de mot de passe
    created_at = Column(DateTime(timezone=True), default=utcnow)  # Date de création (UTC aware)
    last_login = Column(DateTime(timezone=True), nullable=True)  # Dernière connexion (UTC aware)

    presets   = relationship("FilterPreset", back_populates="user", cascade="all,delete-orphan")  # Relation 1-N avec presets
    watchlist = relationship("Watchlist", back_populates="user", cascade="all,delete-orphan")  # Relation 1-N avec watchlist
    history   = relationship("ScanHistory", back_populates="user", cascade="all,delete-orphan")  # Relation 1-N avec historique
    logs      = relationship("ActionLog", back_populates="user", cascade="all,delete-orphan")  # Relation 1-N avec logs d'actions

class FilterPreset(Base):  # Définition du modèle/table "presets"
    __tablename__ = "presets"  # Nom de la table

    id         = Column(Integer, primary_key=True)  # Clé primaire
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # FK vers users.id (+ index)
    name       = Column(String(80), nullable=False)  # Nom du preset
    params_js  = Column(Text, nullable=False)  # JSON  # Contenu JSON sérialisé des paramètres
    created_at = Column(DateTime(timezone=True), default=utcnow)  # Date de création (UTC aware)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)  # Date de MAJ (auto)

    user = relationship("User", back_populates="presets")  # Relation inverse vers User
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_preset_name"),)  # Contrainte d'unicité par (user_id, name)

class Watchlist(Base):  # Définition du modèle/table "watchlist"
    __tablename__ = "watchlist"  # Nom de la table

    id         = Column(Integer, primary_key=True)  # Clé primaire
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # FK vers users.id
    symbol     = Column(String(32), nullable=False)  # Symbole du token
    pair_addr  = Column(String(96), nullable=False)  # pool address (Solana)  # Adresse du pool Dex
    mint       = Column(String(96), nullable=True)  # Adresse mint optionnelle
    notes      = Column(Text, nullable=True)  # Notes libres
    created_at = Column(DateTime(timezone=True), default=utcnow)  # Date d'ajout (UTC aware)

    user = relationship("User", back_populates="watchlist")  # Relation inverse vers User
    __table_args__ = (UniqueConstraint("user_id", "pair_addr", name="uq_user_pair"),)  # Un même pair ne peut exister qu'une fois par user

    # --- alias pour compatibilité avec les tests / API
    @property
    def pair(self) -> str:  # Alias de lecture pour pair_addr
        return self.pair_addr  # Renvoie l'adresse du pair

    @property
    def pair_address(self) -> str:  # Second alias de lecture
        return self.pair_addr  # Renvoie l'adresse du pair

class ScanHistory(Base):  # Définition du modèle/table "scan_history"
    __tablename__ = "scan_history"  # Nom de la table

    id        = Column(Integer, primary_key=True)  # Clé primaire
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # FK vers users.id (+ index)
    ts        = Column(DateTime(timezone=True), default=utcnow, index=True)  # Timestamp du scan (UTC aware + index)
    symbol    = Column(String(32), nullable=False)  # Symbole du token scanné
    pair_addr = Column(String(96), nullable=False)  # Adresse du pool scanné
    score     = Column(Float, nullable=True)  # Score calculé au moment du scan
    risk_prob = Column(Float, nullable=True)  # Probabilité de risque (modèle) au moment du scan
    liq       = Column(Float, nullable=True)  # Liquidité au moment du scan
    buy_press = Column(Float, nullable=True)  # Pression d'achat au moment du scan

    user = relationship("User", back_populates="history")  # Relation inverse vers User

class ActionLog(Base):  # Définition du modèle/table "action_logs"
    __tablename__ = "action_logs"  # Nom de la table

    id      = Column(Integer, primary_key=True)  # Clé primaire
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # FK (optionnelle) vers users.id (+ index)
    ts      = Column(DateTime(timezone=True), default=utcnow, index=True)  # Timestamp du log (UTC aware + index)
    action  = Column(String(64), nullable=False)  # Nom/clé de l'action
    meta_js = Column(Text, default="{}")  # Métadonnées additionnelles en JSON (texte)

    user = relationship("User", back_populates="logs")  # Relation inverse vers User

# ---------------------------------------------------------------------
# Init / helpers
# ---------------------------------------------------------------------
def init_db() -> None:  # Initialise la base : crée les tables si elles n'existent pas
    """Crée les tables si absentes."""
    Base.metadata.create_all(engine)  # Exécute la création des tables via l'engine

def _hash(pwd: str) -> str:  # Hache un mot de passe en PBKDF2-SHA256
    return pbkdf2_sha256.hash(pwd)  # Renvoie la chaîne hachée

def _verify(pwd: str, ph: str) -> bool:  # Vérifie qu'un mot de passe correspond à un hash
    try:
        return pbkdf2_sha256.verify(pwd, ph)  # True si le mot de passe correspond
    except Exception:
        return False  # False si une erreur survient (ex. hash invalide)

# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------
def create_user(username: str, password: str) -> Tuple[bool, str]:  # Crée un utilisateur en base
    username = (username or "").strip()  # Nettoie le username (supprime espaces, gère None)
    if not username or not password:  # Vérifie la présence des champs obligatoires
        return False, "Username et mot de passe requis."  # Erreur si manquants
    if len(username) < 3:  # Vérifie longueur minimale username
        return False, "Username trop court (≥ 3)."  # Erreur si trop court
    if len(password) < 6:  # Vérifie longueur minimale mot de passe
        return False, "Mot de passe trop court (≥ 6)."  # Erreur si trop court

    with SessionLocal() as db:  # Ouvre une session DB
        if db.query(User).filter(User.username == username).first():  # Vérifie l'unicité du username
            return False, "Ce username existe déjà."  # Erreur si déjà pris
        u = User(username=username, password_h=_hash(password))  # Construit l'objet User avec mot de passe haché
        db.add(u)  # Ajoute à la session
        db.commit()  # Valide en base
    return True, "Compte créé."  # Succès

def authenticate(username: str, password: str) -> Optional[User]:  # Authentifie un utilisateur
    with SessionLocal() as db:  # Ouvre une session DB
        u = db.query(User).filter(User.username == (username or "").strip()).first()  # Récupère l'utilisateur par username
        if u and _verify(password or "", u.password_h):  # Vérifie le mot de passe
            u.last_login = utcnow()  # Met à jour la date du dernier login
            db.commit()  # Enregistre la mise à jour
            return u  # Renvoie l'objet User authentifié
    return None  # Échec d'authentification

# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------
def save_preset(user_id: int, name: str, params) -> tuple[bool, str]:  # Sauvegarde/MAJ d'un preset utilisateur
    """
    Les tests appellent: save_preset(user_id, name, json.dumps(params))
    -> on accepte dict OU str et on renvoie (ok, msg).
    """
    try:
        js = params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)  # Sérialise en JSON si dict
        with SessionLocal() as db:  # Ouvre une session
            p = db.query(FilterPreset).filter_by(user_id=user_id, name=name).first()  # Cherche un preset existant
            if p:  # Si trouvé, met à jour
                p.params_js = js  # Met à jour le JSON
                p.updated_at = utcnow()  # Met à jour le timestamp
            else:  # Sinon, crée un nouveau preset
                db.add(FilterPreset(user_id=user_id, name=name, params_js=js))  # Ajoute le nouvel objet
            db.commit()  # Valide en base
        return True, "Preset enregistré"  # Succès
    except Exception as e:  # Capture des erreurs
        return False, f"Erreur preset: {e}"  # Retourne l'erreur formatée

def list_presets(user_id: int):  # Liste les presets d'un utilisateur
    """
    Les tests s'attendent à itérer sur des objets avec un attribut .name
    -> on renvoie les objets ORM directement.
    """  # Docstring
    with SessionLocal() as db:  # Ouvre une session
        return (
            db.query(FilterPreset)
              .filter_by(user_id=user_id)  # Filtre par utilisateur
              .order_by(FilterPreset.updated_at.desc())  # Trie par dernière mise à jour décroissante
              .all()  # Récupère la liste complète
        )

# ---------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------
def add_watchlist(  # Ajoute un élément à la watchlist utilisateur
    user_id: int,
    pair_addr: str,
    symbol: str,
    mint: Optional[str] = None,
    notes: str = "",
) -> tuple[bool, str]:
    """
    Les tests appellent: add_watchlist(user_id, pair_addr, symbol)
    -> on met pair_addr en 2e param et mint devient optionnel.
    """
    try:
        with SessionLocal() as db:  # Ouvre une session
            if db.query(Watchlist).filter_by(user_id=user_id, pair_addr=pair_addr).first():  # Vérifie doublon
                return False, "Déjà présent dans ta watchlist."  # Refuse si déjà existant
            w = Watchlist(  # Crée un nouvel objet Watchlist
                user_id=user_id,
                pair_addr=pair_addr or "",
                symbol=symbol or "?",
                mint=mint,
                notes=notes or "",
            )
            db.add(w)  # Ajoute à la session
            db.commit()  # Valide en base
        return True, "Ajouté à ta watchlist."  # Succès
    except Exception as e:  # Capture des erreurs
        return False, f"Erreur watchlist: {e}"  # Retourne l'erreur

def list_watchlist(user_id: int) -> List[Watchlist]:  # Retourne la watchlist d'un utilisateur
    with SessionLocal() as db:  # Ouvre une session
        return (
            db.query(Watchlist)
              .filter_by(user_id=user_id)  # Filtre par utilisateur
              .order_by(Watchlist.created_at.desc())  # Trie par date d'ajout décroissante
              .all()  # Renvoie la liste d'objets ORM
        )

# ---------------------------------------------------------------------
# Historique de scan
# ---------------------------------------------------------------------
def save_scan_rows(  # Sauvegarde en base une liste de résultats de scan
    user_id: int,
    rows: List[Dict[str, Any]],
    risk_map: Optional[Dict[str, float]] = None,
) -> tuple[int, str]:
    """
    Les tests appellent: n_saved, msg = save_scan_rows(user_id, rows)
    -> risk_map devient optionnel et on renvoie (n, msg).
    """
    risk_map = risk_map or {}  # Utilise un dict vide si non fourni
    count = 0  # Compteur de lignes sauvegardées
    now = utcnow()  # Timestamp commun pour toutes les entrées
    with SessionLocal() as db:  # Ouvre une session
        for m in rows:  # Parcourt chaque métrique scannée
            db.add(  # Prépare l'insertion d'une ligne d'historique
                ScanHistory(
                    user_id=user_id,
                    ts=now,
                    symbol=m.get("symbol") or "?",
                    pair_addr=m.get("address") or "",
                    score=float(m.get("_score") or 0),
                    risk_prob=float(risk_map.get(m.get("address", ""), 0) or 0),
                    liq=float(m.get("liq") or 0),
                    buy_press=float(m.get("buy_pressure") or 0),
                )
            )
            count += 1  # Incrémente le nombre d'insertions
        db.commit()  # Valide toutes les insertions en une fois
    return count, f"{count} lignes sauvegardées"  # Retourne le total et un message

def list_history(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:  # Récupère l'historique des scans d'un user
    with SessionLocal() as db:  # Ouvre une session
        rows = (
            db.query(ScanHistory)
              .filter_by(user_id=user_id)  # Filtre par utilisateur
              .order_by(ScanHistory.ts.desc())  # Trie par date décroissante
              .limit(limit)  # Limite le nombre de résultats
              .all()  # Exécute la requête
        )
        return [  # Transforme chaque objet ORM en dict simple
            {
                "ts": r.ts,
                "symbol": r.symbol,
                "pair": r.pair_addr,
                "score": r.score,
                "risk_prob": r.risk_prob,
                "liq": r.liq,
                "buy_pressure": r.buy_press,
            }
            for r in rows
        ]

# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------
def log_action(user_id: Optional[int], action: str, meta: Optional[Dict[str, Any]] = None) -> None:  # Logge une action
    with SessionLocal() as db:  # Ouvre une session
        db.add(ActionLog(user_id=user_id, action=action, meta_js=json.dumps(meta or {})))  # Ajoute le log (meta JSON)
        db.commit()  # Valide en base

# ---------------------------------------------------------------------
# Export public
# ---------------------------------------------------------------------
__all__ = [  # Exporte explicitement les symboles publics du module
    "init_db",
    "create_user", "authenticate",
    "save_preset", "list_presets",
    "add_watchlist", "list_watchlist",
    "save_scan_rows", "list_history",
    "log_action",
    "User", "FilterPreset", "Watchlist", "ScanHistory", "ActionLog",
    "SessionLocal", "engine", "Base",
]
