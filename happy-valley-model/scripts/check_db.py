import sqlite3, pandas as pd

conn = sqlite3.connect("data/historical/hkjc.db")

print("Races:", pd.read_sql("SELECT COUNT(*) FROM races", conn))
print("Runners:", pd.read_sql("SELECT COUNT(*) FROM runners", conn).iloc[0,0])
print("Results:", pd.read_sql("SELECT COUNT(*) FROM results", conn).iloc[0,0])

df = pd.read_sql("SELECT * FROM races LIMIT 5", conn)
print(df)

conn.close()
