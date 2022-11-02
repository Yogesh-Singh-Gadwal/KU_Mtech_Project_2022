import pandas as pd
import numpy as np
import re
import emot
import emoji
from bs4 import BeautifulSoup
import contractions
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from spellchecker import SpellChecker
# multi processing
import time
import multiprocessing
from multiprocessing import Pool,current_process
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


lcnt=0


# converting emoticon to text using emot package
def emoticon_to_text(text):
  res=""
  emot_obj = emot.core.emot()
  for word in text.split():
    emoticons_dict=emot_obj.emoticons(word)
    if emoticons_dict['flag']:
      mean=emoticons_dict['mean'][0].split()
      word="_".join(mean)
    res=res+word+" "
  return res


# reference https://stackoverflow.com/a/14603508
def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)


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



def stop_words_remover(text):
  stop = stopwords.words('english')
  return " ".join(x for x in text.split() if x not in stop and len(x) > 3)


def remove_freqwords(text):
  cnt = Counter()
  for word in text.split():
    cnt[word] += 1
  cnt.most_common(5)
  FREQWORDS = set([w for (w, wc) in cnt.most_common(5)])
  return " ".join([word for word in str(text).split() if word not in FREQWORDS])


def remove_rarewords(text):
    cnt = Counter()
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def apply_all_text_cleaners(text):
    global lcnt

    if lcnt % 10000 == 0:
        print("--Cleaning {}th review--".format(lcnt))
    lcnt+=1
    try:
      text = BeautifulSoup(text, 'html.parser').get_text() #parse html
      text = emoji.demojize(text, delimiters=("", "")) # emoji to text
    # text = emoticon_to_text(text) #emoticons to text
      text = contractions.fix(text) # replace contractions: i'll to i will
      text = text.lower() # normalising text to lower case
    # text = remove_text_inside_brackets(text) 
      text = re.sub('[^a-zA-Z]', ' ', text) # Removing Punctuations and Numbers
      text = stop_words_remover(text)
      text = re.sub(r"\s+[a-zA-Z]\s+", '', text) # Single character removal
      text = remove_freqwords(text)
    # text= remove_rarewords(text)
      text= lemmatize_words(text)
    finally:
      return text


if __name__ == '__main__':

  t1=time.time()
  df = pd.read_csv('output/Amazon_reviews_verified_f1.csv')
  # create and configure the process pool
  with Pool() as pool:
    # execute tasks in order
    res = pool.map(apply_all_text_cleaners, df['review'], chunksize=1024)
    # process pool is closed automatically

  
  print("pool took:",time.time()-t1)
  # print(res)
  # df['text'] = [x for x in res]
  df.loc[:,'text'] = res
  df 
  df.to_csv('output/Amazon_reviews_cleaned_f1.csv', index=False)


