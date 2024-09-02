import sqlite3


class SQLite3Database:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

        self.create_table(
            "users",
            '''
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            surname TEXT NOT NULL,
            passphrase TEXT NOT NULL,
            faiss_index_id INTEGER NOT NULL
            '''
        )

    def create_table(self, table_name, columns):
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute(query)

    def execute(self, query, fetch=False):
        self.cursor.execute(query)
        self.connection.commit()

        if fetch:
            return self.cursor.fetchall()

        return None

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def close(self):
        self.connection.close()

    def __del__(self):
        self.connection.close()
