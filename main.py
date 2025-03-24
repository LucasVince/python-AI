import pandas as pd
import re
import sklearn
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes

df = pd.read_csv('./phrases.csv')

def clear_text(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

df['phrase'] = df['phrase'].apply(clear_text)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
x = vectorizer.fit_transform(df['phrase'])

# print(df.head)
# print(frases_vetorizadas.toarray())

y = df['class']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

model = sklearn.naive_bayes.MultinomialNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred)}')

def predict_phrase(phrase):
    phrase = clear_text(phrase)
    vectorized_phrase = vectorizer.transform([phrase])
    predict = model.predict(vectorized_phrase)
    return predict

new_phrase = input('type a phrase(negative or positive), and the AI will try to predict: ')
res = predict_phrase(new_phrase)
print(res)
store_phrase = input('is it right??')

if store_phrase.lower() == 'yes':
    pandas_new_phrase = {
        'phrase': f'{new_phrase}',
        'class': f'{res[0]}'
    }

    df_new_phrase = pd.DataFrame([pandas_new_phrase])

    df_new_phrase.to_csv('./phrases.csv', mode='a', header=False, index=False)
else:
    print('thanks for the feed back, will try to inproove')