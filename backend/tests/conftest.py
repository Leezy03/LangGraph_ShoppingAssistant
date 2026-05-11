import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import ShoppingRequest
from tests.helpers import build_test_advisor


@pytest.fixture
def shopping_request():
    return ShoppingRequest(product_name="手机", concerns=["拍照"])


@pytest.fixture
def advisor_factory():
    return build_test_advisor
