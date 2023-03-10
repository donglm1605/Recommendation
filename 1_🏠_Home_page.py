import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import regex
from datetime import datetime
from underthesea import word_tokenize
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import joblib
import re

from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords, preprocess_string

# GUI
st.set_page_config(page_title="Recommendation System", 
                   page_icon= "✍" )
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("https://i.pinimg.com/564x/c3/3d/ba/c33dbacd88f43b1db3abc7c0ed5e6332.jpg");
        background-size: auto;
            }}
</style>
    """, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: #48D1CC;'>Data Science Project</h1>",
            unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #fff375;'>RECOMMENDATION SYSTEM FOR TIKI.VN</h1>",
            unsafe_allow_html=True)

audio_file = open("audio.ogg", "rb")
audio_bytes = audio_file.read()

st.audio(audio_bytes, format="audio/ogg")