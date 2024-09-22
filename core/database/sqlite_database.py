import sqlite3
import shortuuid

from typing import Tuple
from dataclasses import dataclass


@dataclass
class User:
    """
    Dataclass for user

    Attributes:
        user_id (str): user id
        passphrase (str): user passphrase
        faiss_index (int): faiss index id
    """
    user_id: str
    passphrase: str
    faiss_index: int


class SQLiteDatabase:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        self.create_table(
            "users",
            '''
            user_id TEXT PRIMARY KEY,
            passphrase TEXT NOT NULL,
            faiss_index_id INTEGER NOT NULL
            '''
        )

    def create_table(self, table_name: str, columns: str):
        """
        Create table in SQLite3 database

        Args:
            table_name (str): name of table
            columns (str): columns to create            
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"

        self.execute(query)

    def execute(self, query: str, params: Tuple = None, fetch: bool = False):
        """
        Execute SQLite3 query

        Args:
            query (str): SQLite3 query
            fetch (bool): whether to fetch results

        Returns:
            list: results of query if fetch is True
        """
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        
        self.connection.commit()

        if fetch:
            return self.cursor.fetchall()

        return None

    def fetchall(self):
        """
        Fetch all results from last query

        Returns:
            list: results of last query
        """
        return self.cursor.fetchall()

    def fetchone(self):
        """
        Fetch one result from last query

        Returns:
            tuple: result of last query
        """
        return self.cursor.fetchone()

    def close(self):
        """
        Close SQLite3 connection
        """
        self.connection.close()

    def __del__(self):
        """
        Close SQLite3 connection when object is deleted
        """
        self.connection.close()


class UserDatabase(SQLiteDatabase):
    def __init__(self, db_path: str = "users.db"):
        super().__init__(db_path)

    def insert_user(self, passphrase: str, faiss_index_id: int):
        """
        Insert user into SQLite3 database

        Args:
            passphrase (str): user passphrase
            faiss_index_id (int): faiss index id
        """
        user_id = shortuuid.ShortUUID().random(length=7)

        query = f"""
        INSERT INTO users (user_id, passphrase, faiss_index_id)
        VALUES (?, ?, ?)
        """

        self.execute(query, params=(user_id, passphrase, faiss_index_id))

        return user_id 
    
    def get_user_by_id(self, user_id: str):
        """
        Fetch a user from the database by their user_id.

        Args:
            user_id (str): The user_id to search for.

        Returns:
            tuple: The user record, or None if no user is found.
        """
        query = "SELECT * FROM users WHERE user_id = ?"
        result = self.execute(query, params=(user_id,), fetch=True)

        if result:
            return User(*result[0])
    
    def get_all_users(self):
        """
        Fetch all users from the database.

        Returns:
            list of tuples: All user records.
        """
        query = "SELECT * FROM users"

        users = self.execute(query, fetch=True)

        return [User(*user) for user in users]
