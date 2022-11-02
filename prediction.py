# General Imports
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import contractions
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import streamlit as st

def stop_words_remover(text):
  stop = stopwords.words('english')
  return " ".join(x for x in text.split() if x not in stop and len(x) > 3)

def correct_spellings(text):
  spell = SpellChecker()
  corrected_text = []
  misspelled_words = spell.unknown(text.split())
  for word in text.split():
      if word in misspelled_words:
          corrected_text.append(spell.correction(word))
      else:
          corrected_text.append(word)
  return " ".join(corrected_text)

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Text cleaner
@st.cache
def cleaner(text):
    text = emoji.demojize(text, delimiters=("", "")) # emoji to text
    text = contractions.fix(text) # replace contractions: i'll to i will
    text = text.lower() # normalising text to lower case
    text = correct_spellings(text) # normalising text to lower case
    text = re.sub('[^a-zA-Z]', ' ', text) # Removing Punctuations and Numbers
    text = stop_words_remover(text)
    text = re.sub(r"\s+[a-zA-Z]\s+", '', text) # Single character removal
    text= lemmatize_words(text)
    return text

def rating_image(argument):
    switcher = {
        1: "images/1.2.gif",
        2: "images/2.2.gif",
        3: "images/3.2.gif",
        4: "images/4.2.gif",
        5: "images/5.2.gif",
    }
    return switcher.get(argument, " ")

def sentiment_image(argument):
    switcher = {
        "POSITIVE": "images/pos.gif",
        "NEGATIVE": "images/neg.gif",
        "NEUTRAL": "images/neu.gif",
    }
    return switcher.get(argument, " ")

def category_image(argument):
    switcher = {
        "PACKAGING": "images/package2.gif",
        "PRODUCT": "images/product2.gif",
        "DELIVERY": "images/delivery2.gif",
    }
    return switcher.get(argument, " ")

# load the model from disk
global detect_Model
detectFile = open('model/SVC_final.pkl','rb')
detect_Model = pickle.load(detectFile)
detectFile.close()

st.header("Review Prediction leveraging Multi Class Multi Output Classification")

review = '''That book was THE best wrapt book I ever received. Thank you seller for taking so much care to wrap this book! It was in many coats of film, then schock absorbing paper package, sealed with plastic again. Honestly - book would survive any water for a week. Ordered to deliver to Riga, Latvia. Content is excelent and useful, as expected. '''

text = st.text_area("Type or Copy your Review here", review)

clicked = st.button("SUBMIT")

if clicked:
    clean_text=cleaner(text)
    st.subheader("Processed text is: ")
    clean_text
    res = detect_Model.predict([clean_text])
    st.subheader("Predictions: ")
    
    col1, col2, col3 = st.columns(3)
    
    col1.subheader("Rating")
    col1.text(int(float(res[0][0])))
    col1.image(rating_image(int(float(res[0][0]))))
    
    col2.subheader("Sentiment")
    col2.text(res[0][1].upper())
    col2.image(sentiment_image(res[0][1].upper()))
    
    col3.subheader("Category")
    col3.text(res[0][2].upper())
    col3.image(category_image(res[0][2].upper()))