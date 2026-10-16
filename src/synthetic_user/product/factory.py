"""Build the configured ProductClient."""

from __future__ import annotations

from synthetic_user.core.config import AppConfig, ProductClientKind
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.product.base import ProductClient
from synthetic_user.product.fixture_client import FixtureProductClient


def create_product_client(settings: AppConfig) -> ProductClient:
    product = settings.product
    if product.client is ProductClientKind.FIXTURE:
        if product.fixture_path is None:
            raise ConfigError("product.fixture_path is required when product.client is 'fixture'")
        return FixtureProductClient.from_file(product.fixture_path)
    raise ConfigError(f"Unknown product.client: {product.client!r}")
