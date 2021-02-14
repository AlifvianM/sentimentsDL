import re

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = nltk.corpus.stopwords.words(['english'])



lem = WordNetLemmatizer()

def cleaning(data):
  #remove urls
  tweet_without_url = re.sub(r'http\S+',' ', data)

  #remove hashtags
  tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)

  #3. Remove mentions and characters that not in the English alphabets
  tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
  precleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)

    #2. Tokenize
  tweet_tokens = TweetTokenizer().tokenize(precleaned_tweet)
    
    #3. Remove Puncs
  tokens_without_punc = [w for w in tweet_tokens if w.isalpha()]
    
    #4. Removing Stopwords
  tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    
    #5. lemma
  text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]
    
    #6. Joining
  return " ".join(text_cleaned)