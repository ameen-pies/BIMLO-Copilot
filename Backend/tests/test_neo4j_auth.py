import pytest
from neo4j_auth import _hash_password, _verify_password


def test_bcrypt_hash_and_verify():
    pw = "test_password_123"
    hashed = _hash_password(pw)
    assert hashed != pw
    assert hashed.startswith("$2b$")
    assert _verify_password(pw, hashed) is True


def test_bcrypt_wrong_password():
    pw = "correct_password"
    wrong = "wrong_password"
    hashed = _hash_password(pw)
    assert _verify_password(wrong, hashed) is False


def test_sha256_legacy_compatibility():
    import hashlib, secrets
    pw = "legacy_password"
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + pw).encode()).hexdigest()
    stored = f"{salt}:{digest}"
    assert _verify_password(pw, stored) is True
