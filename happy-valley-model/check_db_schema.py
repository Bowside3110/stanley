#!/usr/bin/env python3
import sqlite3

# Connect to the database
conn = sqlite3.connect('data/historical/hkjc.db')
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', tables)

# For each table, get its schema
for table in [t[0] for t in tables]:
    print(f'\nTable: {table}')
    cursor.execute(f'PRAGMA table_info({table})')
    columns = cursor.fetchall()
    print(columns)

# Close the connection
conn.close()

