import sqlite3

import pytest


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE people(id INTEGER, name TEXT, score REAL);
            INSERT INTO people VALUES (1,'Ada',1.0),(2,'Bob',2.0),(3,'Ada',NULL);
        """)
    return path
