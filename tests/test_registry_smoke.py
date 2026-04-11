from biosieve.strategies import build_registry


def test_registry_has_exact_reducer():
    reg = build_registry()
    assert reg.get_reducer("exact") is not None
