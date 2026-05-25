"""Tests fuer API-Key-Store Schema-Initialisierung."""

import unittest

import pymysql

from api_key_store import ApiKeyStore


class _Cursor:
    def __init__(self, fail_alter=False):
        self.fail_alter = fail_alter
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if self.fail_alter and "ALTER TABLE api_keys" in sql:
            raise pymysql.err.Error("alter failed")


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class ApiKeyStoreInitDbTests(unittest.TestCase):
    def test_init_db_raises_on_key_prefix_migration_failure(self):
        cursor = _Cursor(fail_alter=True)
        conn = _Connection(cursor)
        store = ApiKeyStore({"host": "x", "user": "u", "password": "p", "database": "d"})
        store._ensure_connection = lambda: setattr(store, "_conn", conn)

        with self.assertRaises(pymysql.err.Error):
            store.init_db()

        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
