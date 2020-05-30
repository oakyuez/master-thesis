import pandas as pd
import sqlite3

# Import NELA 2018 database, convert to csv-file, keep only source and content of the article
con_1 = sqlite3.connect("")

nela = pd.read_sql_query("Select * From articles", con_1)

nela.to_csv("")



