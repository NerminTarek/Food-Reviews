import os
import pickle
import streamlit as st
import sys
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer

model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
model_path = os.path.join('/', model_name)
vect_path = os.path.join('/', vectorizer_name)

model=pickle.dump(clf, open(model_name, 'wb'))
vect=pickle.dump(vectorizer, open(vectorizer_name, 'wb'))


def review_cleaned(review):
    word_list=nltk.word_tokenize(review)
    clean_list=[]
    for word in word_list:
        if word.lower() not in stop :
            stemmed=PorterStemmer().stem(word)
            clean_list.append(stemmed)
    return " ".join(clean_list)    

def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = review_cleaned(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"


def main():
    st.title('Predict the score of review")
    st.markdown("<h1 style='text-align: center; color: White;background-color:#0E1117'>Food Review Classifier</h1>", unsafe_allow_html=True)
    review = st.text_input(label='Write The Review')
    if st.button('Classify'):
        result = raw_test(review, model, vect)
        st.success(
            'This Review Is {}'.format(result))


if __name__ == '__main__':
    main()
