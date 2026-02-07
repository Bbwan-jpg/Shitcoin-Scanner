# tests/conftest.py
import os, sys, importlib
import pytest

# 1) S'assurer que la racine du projet est importable (pour "import app_streamlit", "import db", etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2) Fixture DB isolée pour chaque test (utilise une base SQLite temporaire)
@pytest.fixture(scope="function")
def db_module(tmp_path, monkeypatch):
    # URL pour une base temporaire
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Recharger le module db pour qu'il prenne la nouvelle DATABASE_URL
    import db as db_mod
    importlib.reload(db_mod)

    # Créer les tables
    db_mod.init_db()
    return db_mod

# 3) Deux users de démo pour les tests
@pytest.fixture
def userA(db_module):
    ok, _ = db_module.create_user("alice", "password123")
    assert ok
    user = db_module.authenticate("alice", "password123")
    assert user is not None
    return user

@pytest.fixture
def userB(db_module):
    ok, _ = db_module.create_user("bob", "s3cr3t!")
    assert ok
    user = db_module.authenticate("bob", "s3cr3t!")
    assert user is not None
    return user
