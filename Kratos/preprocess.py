import warnings

from pandas.compat import to_str

warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
# nltk.download('stopwords')
from profanityfilter import ProfanityFilter
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec                                   #For Word2Vec
import pandas as pd

def pre_precess_text(data_sel):
    """
    :param data_sel: selected data from the CSV as pandas dataframe
    :return: no return store output.csv file which is preprocessed
    """
    #removing duplicates
    # final_data = data_sel.drop_duplicates(subset={"TITLE","CATEGORY"})
    stop = set(stopwords.words('english'))

    final_data=data_sel

    #getting text data
    data=final_data['TITLE']
    category=final_data['CATEGORY']
    #for remove greeting data
    Greetings_data = pd.read_csv('../input/greetings_dataset.csv')
    greetings = list(Greetings_data.values.flatten())

    import re
    i=0
    with open('input.csv', 'w') as f:
        f.write("TITLE\n")
    #stemming
        temp = []
        snow = nltk.stem.SnowballStemmer('english')
        for sentence in data:
            sentence = sentence.lower()  # Converting to lowercase
            sentence=re.sub(r'http\S+', '', sentence)
            sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', sentence)
            cleanr = re.compile('<.*?>')
            sentence = re.sub(r'[^\w]', ' ', sentence)
            sentence = re.sub(cleanr, ' ', sentence)  # Removing HTML tags

            sentence = re.sub(r'[^a-zA-Z0-9]', " ", sentence)

            # words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english') if word not in greetings]
            # temp.append(sentence)
            f.write(sentence+"\n")
            i = i + 1

        csv_input = pd.read_csv('input.csv')

        print(len(csv_input))
        print(len(category))
        csv_input['CATEGORY'] = category
        csv_input.to_csv('output.csv')