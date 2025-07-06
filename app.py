import streamlit as st
from joblib import load
import os
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title='Review classifier')
st.title('Review classifier (TF-IDF + SVD + CatBoost)')
st.write('Determines whether the review will be positive or negative')

MODELS_DIR = 'models'
model_files = {
    'vectorizer': 'vectorizer.joblib',
    'svd': 'svd.joblib',
    'model': 'model.joblib'
}

@st.cache_resource
def load_models():
    models = {}
    try:
        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            models[name] = load(path)
        return models
    except Exception as e:
        st.error(f'Error loading models: {e}')
        return None

models = load_models()

user_input = st.text_area('Enter a review for the film:', '', height=150)

if st.button('Analyze review', type="primary"):
    if not user_input.strip():
        st.warning('Enter a review')
    elif models is None:
        st.error('Models not loaded. Check model files.')
    else:
        try:
            tfidf = models['vectorizer']
            svd = models['svd']
            catboost = models['model']
            
            X_tfidf = tfidf.transform([user_input])
            X_svd = svd.transform(X_tfidf)

            prediction = catboost.predict(X_svd)[0]
            proba = catboost.predict_proba(X_svd)[0]

            if prediction == 0:
                st.success('### Positive')
            else:
                st.error('### Negative')

            st.progress(proba[0], text=f'Positive probability : {proba[0]:.1%}')

        except Exception as e:
            st.error(f'Error processing review: {str(e)}')