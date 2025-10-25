# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"test": "data"}


def test_basic():
    """Basic test to verify test setup."""
    assert True


def test_sample_data(sample_data):
    """Test using fixture."""
    assert sample_data["test"] == "data"
