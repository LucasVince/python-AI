import pandas as pd
import re

df = pd.read_csv('./frases.csv')

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

df['frases'] = df['frases'].apply(limpar_texto)

print(df.head)
