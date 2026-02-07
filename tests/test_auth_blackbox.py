# [CAPTURE-TESTS_AUTH]
def test_create_then_login_success(db_module):
    ok, msg = db_module.create_user("charlie", "topsecret")
    # soit True (nouvel utilisateur), soit False si déjà créé sur un run précédent
    user = db_module.authenticate("charlie", "topsecret")
    assert user is not None
    assert getattr(user, "username", None) == "charlie"

def test_login_failure_wrong_password(db_module, userA):
    bad = db_module.authenticate("alice", "WRONG")
    assert bad is None

def test_create_duplicate_username_rejected(db_module, userA):
    ok, msg = db_module.create_user("alice", "another")
    assert ok is False  # on attend un refus pour doublon
