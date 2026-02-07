# [CAPTURE-TESTS_APP_SMOKE]
import importlib
import types

def test_app_imports_without_running_streamlit():
    mod = importlib.import_module("app_streamlit")
    assert isinstance(mod, types.ModuleType)
    # quelques symboles attendus
    assert hasattr(mod, "fetch_pairs")
    assert hasattr(mod, "get_metrics")
