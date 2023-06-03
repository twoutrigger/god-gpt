import pandas as pd
import openai

orig = pd.read_csv("bible_american_standard_version.csv", header=0)

# print(len(orig))
# print(orig.head())

orig = orig[['Book Name', 'Book Number', 'Chapter', 'Text']]
orig[['Book Name', 'Book Number', 'Chapter', 'Text']] = orig[['Book Name', 'Book Number', 'Chapter', 'Text']].astype(str)

# Aggregating biblical text to the chapter level
df = orig.groupby(['Book Name', 'Book Number', 'Chapter'])['Text'].apply(' '.join).reset_index()

df = df[['Text']]

# estimating the number of tokens before generating embedding; roughly 30K tokens
# num_words = pd.Series(' '.join(df['Text']).split()).value_counts()
# print(len(num_words))

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

df['ada_embedding'] = df.Text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

# generate text embedding
# df.to_csv('text_embedding.csv', index=False)
