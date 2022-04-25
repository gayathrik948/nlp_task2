import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import nltk
import string
import re
from nltk.corpus import stopwords
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score, recall_score, precision_score

data = pd.read_excel('1000 leads.xlsx')
df = data.copy()

del df['Lead Name']

df.columns = ['Location', 'Status', 'Status information']

df.dropna(subset = ['Status'], inplace=True)
df['Location'].fillna(df['Location'].mode()[0], inplace = True)
df['Status information'].fillna(df['Status information'].mode()[0], inplace = True)

df["Location"].replace({ 'hyderabad' :  'Hyderabad',  'pune' :  'Pune',
                          'bangalore' :  'Bangalore',   'mumbai'  :  'Mumbai',
                            'chennai' : 'Chennai', 'gurgoan' : 'Gurgaon',
                        'gurgaon' : 'Gurgaon', 'thane' : 'Thane',
                        'aurangabad' : 'Aurangabad', 'vijayawada' : 'Vijayawada',
                        "india":"India"}, inplace=True)

df['Status'].replace({'NOt Converted':'Not Converted', 'Converted ':'Converted','Conveted':'Converted'}, inplace=True)

label_encoder = LabelEncoder()
df['Status']= label_encoder.fit_transform(df['Status'])

df['final'] = df['Status information']

df['final'] = df['final'].str.lower()

def remove_number(text):
  result = re.sub(r'\d+','',text)
  return result
df['final'] = df['final'].apply(remove_number)

def rem_punct(text):
  result = re.sub(r'[^\w\s]', ' ', text)
  return result.strip()
df['final'] = df['final'].apply(rem_punct)

stop_words = set(stopwords.words('english'))
df['final'] = df['final'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

def get_simple_pos(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN

def lemmatize_word(text):
  lemmatizer = WordNetLemmatizer()
  word_tokens = word_tokenize(text)
  output_words =[]
  for word in word_tokens:
    pos=pos_tag([word])
    lemmas = lemmatizer.lemmatize(word, pos=get_simple_pos(pos[0][1]))
    output_words.append(lemmas)
  return output_words

df['final'] = df['final'].apply(lemmatize_word)

final = df['final'].values

final_x = []
for i in range(len(final)):
    final_x.append(final[i])

final_x1 = [" ".join(x) for x in final_x]

df['final'] = final_x1

def count_punt(text):
  count = sum([1 for char in text if char in string.punctuation])
  return round(count/(len(text) - text.count(' ')), 3)*100

df['text_len'] = df['Status information'].apply(lambda x:len(x) - x.count(' '))
df['punct'] = df['Status information'].apply(lambda x:count_punt(x))

vec = TfidfVectorizer(max_features=1100)
final_vec = vec.fit_transform(df['final'])


df1 = df['Location'].to_frame()
df1.reset_index(drop=True, inplace = True)
df1 = pd.get_dummies(df1, drop_first=True)


df2 = df[['text_len', 'punct']]
df2.reset_index(drop=True, inplace = True)

x = pd.concat([df1, df2, pd.DataFrame(final_vec.toarray())], axis=1)
y = df['Status']

x, y = SMOTE().fit_resample(x,y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=100, stratify=y)

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_test)
print(gbc.score(x_train, y_train))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1(y_test, y_pred))



def predict_status(location, text):
    loc_ = [location]
    return loc_

    data = [text]
    return data

    loc = pd.DataFrame(loc_)
    loc.reset_index(drop=True, inplace=True)
    final_loc = pd.get_dummies(loc)

    vect = pd.DataFrame(vec.transform(data).toarray())

    text_len = pd.DataFrame([len(data) - data.count(' ')])
    punct = pd.DataFrame([count_punt(data)])

    new_data = pd.concat([text_len, punct], axis=1)
    new_data.reset_index(drop=True, inplace=True)

    df = pd.concat([final_loc, new_data, vect], axis=1)
    my_pred = gbc.predict(df)
    return my_pred[0]



def main():
  st.title("BEPEC PRODUCT CONVERTED OR NOT-CONVERTED PREDICTION")
  location = st.selectbox('Location', ('Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'delhi', 'australia',
       'nagpur', 'madurai', 'mysore', 'Chennai', 'kerala', 'hubli',
       'guntur', 'jalandhar', 'tiruttani', 'USA', 'Faridabad', 'Online',
       'Gurgaon', 'kochi', 'noida', 'ahmedabad', 'khammam',
       'vishakapatnam', 'solapur', 'nasik', 'Thane', 'UAE', 'Aurangabad',
       'rayagada', 'bilgi', 'Rajamundry', 'kadapa', 'kolkatta',
       'Vijayawada', 'India', 'meerut', 'jaipur', 'coimbatore', 'gujarat',
       'ghazibad', 'ongole', 'Nepal', 'bihar'))
  text = st.text_input("Enter Your Message Here")


  result = ''
  if st.button("Predict"):
    result = predict_status(location, text)

    if result == "1":
      st.info('Customer Not Converted')


    else:
      st.success("Customer Converted")



if __name__=="__main__":
    main()
