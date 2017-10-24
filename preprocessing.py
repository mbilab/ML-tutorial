'''
text = 'Yeah, I know when I compliment her she won\'t believe me \
        And it\'s so sad to think that she don\'t see what I see \
        But every time she asks me do I look okay? \
        I say When I see your face \
        There\'s not a thing that I would change \
        Cause you\'re amazing Just the way you are'
# split by space
print('normal', text.split(), '\n')

# comma will be a word, abbreviation will be strange
from nltk.tokenize import word_tokenize
print('nltk_tokenize', word_tokenize(text), '\n')

# comma and abbreviation will be ignore
from sklearn.feature_extraction.text import CountVectorizer
print('countvectorizer\'s tokenizer', CountVectorizer().build_tokenizer()(text), '\n')

# lower case, comma will be ignore, abbreviation will make sense, but time consuming
from keras.preprocessing.text import text_to_word_sequence
print('keras text to word sequence', text_to_word_sequence(text))
#################### word to vector #######################
data = ['cute', 'beauty', 'cold', 'cold', 'cold', 'hot', 'beauty', 'cute']

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data = le.fit_transform(data) 
print(data)

# one hot - sklearn
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(sparse=False)
onehot_data_sk = one_hot.fit_transform(data.reshape(len(data), 1))
print(onehot_data_sk)

# one hot - keras
from keras.utils import to_categorical
onehot_data_keras = to_categorical(data)
print(onehot_data_keras)
'''
############# word importance ############
data = ['bobo is cute', \
        'bobo is smart', \
        'bobo is humorous']

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X = v.fit_transform(data)
print(v.get_feature_names(), '\n', X.toarray(), '\n')

# tfidf - tf: term frequency, idf: inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
X = v.fit_transform(data)
print(v.get_feature_names(), '\n', X.toarray(), '\n')

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3, svd_solver='full')
print('PCA', pca.fit_transform(X.toarray()), '\n')

# SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=3)
print('SVD', svd.fit_transform(X), '\n')
