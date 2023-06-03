import pandas as pd

orig = pd.read_csv("bible_american_standard_version.csv", header=0)

# print(len(orig))
# print(orig.head())

orig = orig[['Book Name', 'Book Number', 'Chapter', 'Text']]
orig[['Book Name', 'Book Number', 'Chapter', 'Text']] = orig[['Book Name', 'Book Number', 'Chapter', 'Text']].astype(str)

# Aggregating biblical text to the chapter level
df = orig.groupby(['Book Name', 'Book Number', 'Chapter'])['Text'].apply(' '.join).reset_index()

print(len(df))