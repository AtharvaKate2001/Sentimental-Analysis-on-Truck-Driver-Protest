#!/usr/bin/env python
# coding: utf-8

# ## Fetching Data

# ## Connecting with YouTube

# In[1]:


# Import the required libraries
import pandas as pd
from pytube import YouTube
from googleapiclient.discovery import build

# Define a function to get comments from a YouTube video and store them in a DataFrame
def get_all_comments_to_dataframe(video_url, api_key, max_comments=100):
    # Initialize the YouTube API client using the provided API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Extract the video ID from the URL using the pytub library
    video_id = YouTube(video_url).video_id

    # Define a nested function to retrieve comments with pagination
    def get_comments_with_pagination(video_id, max_results=100):
        # Initialize an empty list to store the comments
        comments = []
        # Initialize a variable to track the next page of comments
        nextPageToken = None

        # Continue fetching comments until the desired number is reached
        while len(comments) < max_comments:
            # Call the YouTube API to retrieve comments for the video
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                order="relevance",
                maxResults=min(100, max_comments - len(comments)),
                pageToken=nextPageToken
            ).execute()

            # Extract and append comments from the API response
            for item in results["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                #item["snippet"] accesses the "snippet" section of the comment item, which contains metadata related to the comment.
                comments.append(comment)

            # Check if there are more pages of comments
            if 'nextPageToken' in results:
                #Checks whether the API response (results) contains a "nextPageToken" field.
                #This field is provided by the YouTube Data API when there are additional pages of comments available.
                nextPageToken = results['nextPageToken']
            else:
                # Exit the loop if there are no more comments
                break

        return comments

    # Get all comments for the video using the nested function
    all_comments = get_comments_with_pagination(video_id, max_comments)

    # Create a Pandas DataFrame from the comments, where each comment is a row
    comments_df = pd.DataFrame({'Comment': all_comments})

    # Return the DataFrame containing the comments
    return comments_df


# In[2]:


# pip install pytube


# In[3]:


# Import the required libraries
import pandas as pd
from pytube import YouTube
from googleapiclient.discovery import build

# Define a function to get comments from a YouTube video and store them in a DataFrame
def get_all_comments_to_dataframe(video_url, api_key, max_comments=100):
    # Initialize the YouTube API client using the provided API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Extract the video ID from the URL using the pytub library
    video_id = YouTube(video_url).video_id

    # Define a nested function to retrieve comments with pagination
    def get_comments_with_pagination(video_id, max_results=100):
        # Initialize an empty list to store the comments
        comments = []
        # Initialize a variable to track the next page of comments
        nextPageToken = None

        # Continue fetching comments until the desired number is reached
        while len(comments) < max_comments:
            # Call the YouTube API to retrieve comments for the video
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                order="relevance",
                maxResults=min(100, max_comments - len(comments)),
                pageToken=nextPageToken
            ).execute()

            # Extract and append comments from the API response
            for item in results["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                #item["snippet"] accesses the "snippet" section of the comment item, which contains metadata related to the comment.
                comments.append(comment)

            # Check if there are more pages of comments
            if 'nextPageToken' in results:
                #Checks whether the API response (results) contains a "nextPageToken" field.
                #This field is provided by the YouTube Data API when there are additional pages of comments available.
                nextPageToken = results['nextPageToken']
            else:
                # Exit the loop if there are no more comments
                break

        return comments

    # Get all comments for the video using the nested function
    all_comments = get_comments_with_pagination(video_id, max_comments)

    # Create a Pandas DataFrame from the comments, where each comment is a row
    comments_df = pd.DataFrame({'Comment': all_comments})

    # Return the DataFrame containing the comments
    return comments_df


# In[4]:


# pip install google-api-python-client


# In[5]:


# Import the required libraries
import pandas as pd
from pytube import YouTube
from googleapiclient.discovery import build

# Define a function to get comments from a YouTube video and store them in a DataFrame
def get_all_comments_to_dataframe(video_url, api_key, max_comments=100):
    # Initialize the YouTube API client using the provided API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Extract the video ID from the URL using the pytub library
    video_id = YouTube(video_url).video_id

    # Define a nested function to retrieve comments with pagination
    def get_comments_with_pagination(video_id, max_results=100):
        # Initialize an empty list to store the comments
        comments = []
        # Initialize a variable to track the next page of comments
        nextPageToken = None

        # Continue fetching comments until the desired number is reached
        while len(comments) < max_comments:
            # Call the YouTube API to retrieve comments for the video
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                order="relevance",
                maxResults=min(100, max_comments - len(comments)),
                pageToken=nextPageToken
            ).execute()

            # Extract and append comments from the API response
            for item in results["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                #item["snippet"] accesses the "snippet" section of the comment item, which contains metadata related to the comment.
                comments.append(comment)

            # Check if there are more pages of comments
            if 'nextPageToken' in results:
                #Checks whether the API response (results) contains a "nextPageToken" field.
                #This field is provided by the YouTube Data API when there are additional pages of comments available.
                nextPageToken = results['nextPageToken']
            else:
                # Exit the loop if there are no more comments
                break

        return comments

    # Get all comments for the video using the nested function
    all_comments = get_comments_with_pagination(video_id, max_comments)

    # Create a Pandas DataFrame from the comments, where each comment is a row
    comments_df = pd.DataFrame({'Comment': all_comments})

    # Return the DataFrame containing the comments
    return comments_df


# ## Using API to Fetch Data

# In[6]:


# Replace it with your actual YouTube Data API key
API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Set the YouTube video URL from which you want to retrieve comments
VIDEO_URL = 'https://youtu.be/b81Na6yploU?si=Ar18yKsAyRRO9q4u'

# Set the maximum number of comments you want to retrieve
MAX_COMMENTS = 3500

# Call the function to get comments and create a DataFrame
df = get_all_comments_to_dataframe(VIDEO_URL, API_KEY, MAX_COMMENTS)

# Print the DataFrame containing the comments
print(df)


# In[7]:


# # Specify the path where you want to save the CSV file
# csv_file_path = r'C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv'

# # Save the DataFrame to a CSV file
# df.to_csv(csv_file_path, index=False)

# # Print a message indicating that the CSV file has been saved
# print(f"Comments have been saved to '{csv_file_path}'.")


# In[8]:


# Specify the path where you want to save the CSV file
csv_file_path = r'C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

# Print a message indicating that the CSV file has been saved
print(f"Comments have been saved to '{csv_file_path}'.")


# In[9]:


import warnings
warnings.filterwarnings("ignore")

# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data=pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv")
data.head()


# In[11]:


data=pd.DataFrame(data.Comment)

data.head()


# In[12]:


data.iloc[1,0]  # [row index , column index ]


# In[ ]:





# In[13]:


#import pandas as pd
#from googletrans import Translator


#def translate_hindi_to_english(comment):
#    translator = Translator()
#    try:
#        translation = translator.translate(comment, src='hi', dest='en')
 #       return translation.text
#    except Exception as e:
#        print(f"Error translating: {e}")
#        return comment  # If translation fails, keep the original comment

# Create a new DataFrame with the 'Comment' column translated to English
#data_english = data.copy()
#data_english['Comment'] = data_english['Comment'].apply(translate_hindi_to_english)



# In[ ]:





# In[14]:


# import pandas as pd
# from googletrans import Translator
# from langdetect import detect
# import time



# def translate_hindi_to_english(comment):
#     translator = Translator()
#     max_retries = 5

#     # Detect the language of the comment
#     comment_language = detect(comment)

#     # Translate only if the detected language is Hindi ('hi')
#     if comment_language == 'hi':
#         for _ in range(max_retries):
#             try:
#                 translation = translator.translate(comment, src='hi', dest='en')
#                 return translation.text
#             except Exception as e:
#                 print(f"Error translating: {e}")
#                 time.sleep(2)  # Add a delay before retrying

#     return comment  # If the language is not Hindi or translation fails, keep the original comment

# # Create a new DataFrame with the 'Comment' column translated to English for Hindi comments
# data_english = data.copy()
# data_english['Comment'] = data_english['Comment'].apply(translate_hindi_to_english)







# In[15]:


# pip install langid


# In[16]:


# pip install langdetect


# ## Language Transformation

# In[17]:


# 
import pandas as pd
from googletrans import Translator
import langid
import time

# Assuming 'data' is your DataFrame with a 'Comment' column containing mixed-language comments
# Replace 'your_dataset.csv' with the actual path to your CSV file
data = pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv")

def translate_hindi_to_english(comment):
    translator = Translator()
    max_retries = 5

    # Detect the language of the comment using langid
    lang, _ = langid.classify(comment)

    # Translate only if the detected language is Hindi ('hi')
    if lang == 'hi':
        for _ in range(max_retries):
            try:
                translation = translator.translate(comment, src='hi', dest='en')
                return translation.text
            except Exception as e:
                print(f"Error translating: {e}")
                time.sleep(2)  # Add a delay before retrying

    return comment  # If the language is not Hindi or translation fails, keep the original comment

# Create a new DataFrame with the 'Comment' column translated to English for Hindi comments
data_english = data.copy()
data_english['Comment'] = data_english['Comment'].apply(translate_hindi_to_english)



# In[ ]:





# In[18]:


# #
# import pandas as pd
# from googletrans import Translator
# import langid
# import concurrent.futures
# import time

# # Assuming 'data' is your DataFrame with a 'Comment' column containing mixed-language comments
# # Replace 'your_dataset.csv' with the actual path to your CSV file
# data = pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv")

# def translate_comment(comment):
#     translator = Translator()
    
#     # Detect the language of the comment using langid
#     lang, _ = langid.classify(comment)
    
#     # Translate only if the detected language is Hindi ('hi')
#     if lang == 'hi':
#         max_retries = 5
#         for _ in range(max_retries):
#             try:
#                 translation = translator.translate(comment, src='hi', dest='en')
#                 translated_text = translation.text
#                 if translated_text is not None:
#                     return translated_text
#             except Exception as e:
#                 print(f"Error translating: {e}")
#                 time.sleep(2)  # Add a delay before retrying

#     return comment  # If the language is not Hindi or translation fails, keep the original comment

# def parallel_translate_comments(comments):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         translated_comments = list(executor.map(translate_comment, comments))
#     return translated_comments

# # Split the comments into smaller chunks for parallel processing
# chunk_size = 1000
# num_chunks = len(data) // chunk_size + 1

# # Translate comments in parallel
# translated_comments_list = []
# for i in range(num_chunks):
#     start_idx = i * chunk_size
#     end_idx = (i + 1) * chunk_size
#     chunk_comments = data['Comment'].iloc[start_idx:end_idx]
    
#     translated_chunk = parallel_translate_comments(chunk_comments)
#     translated_comments_list.extend(translated_chunk)

# # Create a new DataFrame with the 'Comment' column translated to English for Hindi comments
# data_english = data.copy()
# data_english['Comment'] = translated_comments_list




# In[19]:


import pandas as pd
from googletrans import Translator
import langid
import time

# Assuming 'data' is your DataFrame with a 'Comment' column containing mixed-language comments
# Replace 'your_dataset.csv' with the actual path to your CSV file
data = pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2.csv")

def translate_comment(comment):
    translator = Translator()

    # Detect the language of the comment using langid
    lang, _ = langid.classify(comment)

    # Translate only if the detected language is not English ('en')
    if lang != 'en':
        max_retries = 5
        for _ in range(max_retries):
            try:
                translation = translator.translate(comment, dest='en')
                translated_text = translation.text
                if translated_text is not None:
                    return translated_text
            except Exception as e:
                print(f"Error translating: {e}")
                time.sleep(2)  # Add a delay before retrying

    return comment  # If the language is already English or translation fails, keep the original comment

# Apply the translation function to the 'Comment' column
data_english = data.copy()
data_english['Comment'] = data_english['Comment'].apply(translate_comment)



# In[20]:


data_english.head


# In[21]:


# Save the new DataFrame to a new CSV file
data_english.to_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2_english.csv", index=False)


# In[ ]:





# # Data Cleaning or Data Pre-Processing

# ## Lower Case

# In[22]:


data_english.Comment = data_english.Comment.str.lower()


# In[23]:


data_english.iloc[1,0]


# ## Remove URL 

# In[24]:


data_english.Comment = data_english.Comment.str.replace(r'http\S+|www.\S+','',case=False)  # | = or 
data_english.iloc[564,0]


# ## Remove Emojis

# In[25]:


# pip install --upgrade emoji


# In[26]:


# import pandas as pd
# import emoji

# #df = pd.read_csv('your_dataset.csv')

# def remove_emojis(Comment):
#     return emoji.get_emoji_regexp().sub(u'', Comment)

# # Apply the remove_emojis function to the 'Comment' column
# data['Comment'] = data['Comment'].apply(remove_emojis)



# In[27]:


# import pandas as pd
# import emoji


# def remove_emojis(comment):
#     emoji_pattern = emoji.get_emoji_regexp()
#     return emoji_pattern.sub(u'', comment)

# # Apply the remove_emojis function to the 'Comment' column
# data['Comment'] = data['Comment'].apply(remove_emojis)



# In[28]:


# import pandas as pd
# import emoji


# def remove_emojis(comment):
#     emoji_pattern = emoji.get_emoji_regexp()
#     return emoji_pattern.sub('', comment)

# # Apply the remove_emojis function to the 'Comment' column
# data_english['Comment'] = data_english['Comment'].apply(remove_emojis)



# In[29]:


# pip install demoji


# In[30]:


import pandas as pd
import demoji


# Download the demoji library's emoji data (needs to be done once)
demoji.download_codes()

def remove_emojis(comment):
    return demoji.replace(comment, '')

# Apply the remove_emojis function to the 'Comment' column
data_english['Comment'] = data_english['Comment'].apply(remove_emojis)



# In[31]:


data_english.iloc[564,0]


# ## Remove Punctuation

# In[34]:


data_english.Comment = data_english.Comment.str.replace(r'http\S+|www.\S+','',case=False)  # | = or 
data_english.iloc[4,0]


# In[35]:


import string
string.punctuation


# In[36]:


def remove_punctuation(abc):
    text_nopunt="".join([c  
                         for c in abc  
                         if c not in string.punctuation])
    return text_nopunt


# In[37]:


data_english.iloc[564,0]


# In[38]:


data_english.Comment = data_english.Comment.apply(lambda x : remove_punctuation(x))  # x = Text 


# In[39]:


data_english.iloc[564,0]


# ## Remove Number

# In[40]:


data_english['Comment'] = data_english['Comment'].str.replace('\d+','')  # data.text  or data['text']


# In[41]:


data_english.iloc[564,0]


# ## Remove StopWords

# In[42]:


# Import stopwords with nltk.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop


# In[43]:


data_english.iloc[1,0]


# In[ ]:


Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
data_english['Comment'] = data_english['Comment'].apply(lambda x: ' '.join([abc   # x = Text 
                                                      for abc in x.split() # word tokenizer # abc = Delhi
                                                      if abc not in (stop)]))


# In[45]:


data_english.iloc[1,0]


# In[ ]:





# ## Common Words

# In[46]:


import re # replace of words
nltk.download('words') # downloading dictionary of nltk  
words = set(nltk.corpus.words.words()) # corpus & set array 
words


# In[47]:


# Apply a second round of cleaning
def clean_text_round2(Comment):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    Comment =  re.sub('delhipolicetorturing|delhipolice|police|delhi', '', Comment) #remove delhipolice
    Comment =  re.sub('driver|drivers|drive|drived','', Comment) #remove farmers
    Comment =  re.sub('protests|protest','', Comment) #remove Protest
    Comment =  re.sub('Modi|modi|MODI','', Comment) #remove Protest
    Comment =  re.sub('Government|goverment|govt|GOVT|government','', Comment) #remove Protest
    Comment =  re.sub('Congress|congress','', Comment) #remove Protest
    Comment =  re.sub('BJP|bjp','', Comment) #remove Protest
    Comment =  re.sub('Sarkar|sarkar|sakaar','', Comment) #remove Protest
    Comment =  re.sub('LOG|log|Log','', Comment) #remove Protest
    Comment =  re.sub('par|Par','', Comment) #remove Protest
    Comment =  re.sub('Hain|hain','', Comment) #remove Protest
    Comment =  re.sub('bhai|Bhai','', Comment) #remove Protest
    Comment =  re.sub('koi|KOI|koi','', Comment) #remove Protest
    Comment =  re.sub('Sab|SAB|sab','', Comment) #remove Protest
    Comment =  re.sub('fir|Fir','', Comment) #remove Protest
    Comment =  re.sub('agar|Agar','', Comment) #remove Protest
    Comment =  re.sub('Hoga|hoga','', Comment) #remove Protest
    Comment =  re.sub('Magar|magar|magr','', Comment) #remove Protest
    Comment =  re.sub('Jay|jay|jai|Jai','', Comment) #remove Protest
    Comment =  re.sub('mein|Mein|MEIN|Main|main','', Comment) #remove Protest
    
    Comment =  re.sub(r"\b[a-zA-Z]\b", "", Comment) ## 1 alphabet like a or s
    Comment =  re.sub(r"\b[a-zA-Z][a-zA-Z]\b", "", Comment)  ## 2 alphabet like ab or ad & aA
    Comment =  " ".join(w 
                     for w in nltk.wordpunct_tokenize(Comment)  ## this will give you tokens 
                      if w.lower() in words)  #    
    return Comment


# In[48]:


data_english = pd.DataFrame(data_english.Comment.apply(lambda x: clean_text_round2(x)))
data_english.iloc[1,0]


# ## Lemmatization

# In[49]:


from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('wordnet') # DOWNLAOD WORDNET

lemmatizer = WordNetLemmatizer() # ASSIGNING
word_tokenizer = nltk.tokenize.WhitespaceTokenizer() ## Word Token


# In[50]:


def lemmatize_text(Comment):
    return [lemmatizer.lemmatize(w,"v") 
            for w in word_tokenizer.tokenize(Comment)]


# In[51]:


data_english.iloc[1,0]


# In[52]:


data_english.Comment = data_english.Comment.apply(lambda x :' '.join(lemmatize_text(x)))
data_english.iloc[1,0]


# ## Remove Extra WhiteSpace

# In[53]:


data_english['Comment'] = (data_english['Comment'].astype("str").str.rstrip())
data_english.iloc[1,0]


# ## Remove Duplicate Row

# In[54]:


data_english = data_english.drop_duplicates('Comment') 
data_english


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TERM- DOCUMENT MATRIX

# In[55]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_english.Comment)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names_out()) # TERM = KEY WORDS

tdm = data_dtm.transpose()
tdm.head()


# In[56]:


tdm['freq'] = tdm.sum(axis=1)
tdm.reset_index(inplace=True)
tdm.head()


# In[57]:


tdm1 = tdm[["index","freq"]] #SUBSET OF 2 COLUMNS
tdm1.rename(columns = {'index':'Word'}, inplace = True) # RENAMING 
tdm1.sort_values(by='freq',ascending=False,inplace=True) # SORTING DATA 
tdm1.head(20)


# # Word Cloud

# In[58]:


Comment = " ".join(review for review in data_english.Comment)
print ("There are {} words in the combination of all review.".format(len(Comment)))


# In[59]:


## collocations=False means try to contro the duplicate keyword and counts as 1
# lower max_font_size, change the maximum number of word and lighten the background:

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)
# Generate a word cloud image
cloud = WordCloud(stopwords=stopwords, # BACK UP
                      background_color="white",
                      collocations=False,
                      max_words=1000).generate(Comment)

# Display the generated image:
# the matplotlib way:
plt.imshow(cloud, interpolation='bilinear') # IM = IMAGE 
plt.axis("off") # NO AXIS
plt.show() # DISPLAY PLOT


# # Sentimental Analysis

# In[60]:


from textblob import TextBlob
data_english['polarity'] = data_english['Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[61]:


data_english['polarity']=data_english['polarity']


# In[62]:


from textblob import TextBlob

# data_english['polarity'] = data_english['Comment'].apply(lambda x: TextBlob(x).sentiment.polarity) # polarity range -1 to +1
# data_english.head(5)


# In[63]:


data_english['Sentiment'] = np.where(data_english['polarity'] >= 0, 'Positive', 'Negative')


# In[64]:


data_english['Sentiment']=data_english['Sentiment']


# In[65]:


# data_english['Sentiment'] = np.where(data_english['polarity']>= 0, 'Positive', 'Negative')
# data_english.head()


# In[66]:


data_english.Sentiment.value_counts().plot.pie(autopct="%.1f%%");


# In[ ]:





# # TF-IDF

# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer 
 
# settings that you use for count vectorizer will go here 
tfidf_vectorizer=TfidfVectorizer(use_idf=True,stop_words='english',analyzer='word') 
 
# just send in all your docs here 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(data_english.Comment)
feature_names = cv.get_feature_names_out()  ## EVERY WORD NAME
abc=tfidf_vectorizer_vectors.transpose()


# In[68]:


output=pd.DataFrame.sparse.from_spmatrix(abc,index=feature_names)
output.reset_index(inplace=True)
output.rename(columns = {'index':'Word'}, inplace = True) # RENAMING
output


# In[69]:


output['TF*IDF'] = output.max(axis=1)
output=output[["Word","TF*IDF"]] #SUBSET OF 2 COLUMNS
output.sort_values(by='TF*IDF',ascending=False,inplace=True) # SORTING DATA 
output.head(20)


# # Saving cleaned data

# In[70]:


# data_english.to_csv('capstone_project2.csv', index=False)


# In[71]:


# Save the new DataFrame to a new CSV file
data_english.to_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2_withterms.csv", index=False)


# In[72]:


print(data_english.columns)


# In[73]:


df=pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2_withterms.csv")


# In[74]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Converting words to categorical values

# In[75]:


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()


# In[76]:


# le.fit(df['Comment'])
# df['Comment'] = le.transform(df['Comment'])


# In[77]:


# df.Comment.unique()


# In[78]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(stop_words='english')

# tfidf.fit(df['Sentiment'])
# requredTaxt  = tfidf.transform(df['Sentiment'])


# In[79]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Comment'], test_size=0.2, random_state=42)


# In[80]:


# X_train.shape


# In[81]:


# X_test.shape


# # Model Building

# In[82]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your preprocessed dataset
# Assuming you have a DataFrame with 'text' as the column containing comments and 'label' as the column with sentiment labels
df = pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2_withterms.csv")

df['Comment'].fillna('', inplace=True)
# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['Comment'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
train_features = tfidf_vectorizer.fit_transform(train_data)
test_features = tfidf_vectorizer.transform(test_data)

# ... (your previous code)

# Train and evaluate Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(train_features, train_labels)
logistic_regression_train_predictions = logistic_regression_model.predict(train_features)
logistic_regression_test_predictions = logistic_regression_model.predict(test_features)

logistic_regression_train_accuracy = accuracy_score(train_labels, logistic_regression_train_predictions)
logistic_regression_test_accuracy = accuracy_score(test_labels, logistic_regression_test_predictions)

# Train and evaluate Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_features, train_labels)
naive_bayes_train_predictions = naive_bayes_model.predict(train_features)
naive_bayes_test_predictions = naive_bayes_model.predict(test_features)

naive_bayes_train_accuracy = accuracy_score(train_labels, naive_bayes_train_predictions)
naive_bayes_test_accuracy = accuracy_score(test_labels, naive_bayes_test_predictions)

# Train and evaluate Random Forest model
# Train and evaluate Random Forest model with adjusted hyperparameters
random_forest_model = RandomForestClassifier(
    n_estimators=100,  # You can adjust the number of trees
    max_depth=20,       # You can adjust the maximum depth of each tree
    min_samples_split=2,  # You can adjust the minimum number of samples required to split an internal node
    min_samples_leaf=1   # You can adjust the minimum number of samples required to be at a leaf node
)

random_forest_model.fit(train_features, train_labels)
random_forest_train_predictions = random_forest_model.predict(train_features)
random_forest_test_predictions = random_forest_model.predict(test_features)

random_forest_train_accuracy = accuracy_score(train_labels, random_forest_train_predictions)
random_forest_test_accuracy = accuracy_score(test_labels, random_forest_test_predictions)

# Print accuracy for each model
print(f"Logistic Regression Train Accuracy: {logistic_regression_train_accuracy:.2f}")
print(f"Logistic Regression Test Accuracy: {logistic_regression_test_accuracy:.2f}")

print(f"Naive Bayes Train Accuracy: {naive_bayes_train_accuracy:.2f}")
print(f"Naive Bayes Test Accuracy: {naive_bayes_test_accuracy:.2f}")

print(f"Random Forest Train Accuracy: {random_forest_train_accuracy:.2f}")
print(f"Random Forest Test Accuracy: {random_forest_test_accuracy:.2f}")



# In[83]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
# The dataset should have a 'text' column containing the text data and a 'label' column with sentiment labels (e.g., 'positive' or 'negative').
df = pd.read_csv(r"C:\Users\Atharva\OneDrive\Desktop\Capstone 2\capstone_project2_withterms.csv")

# Handle missing values in the 'Comment' column by replacing NaN with an empty string
df['Comment'].fillna('', inplace=True)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['Comment'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
train_features = tfidf_vectorizer.fit_transform(train_data)
test_features = tfidf_vectorizer.transform(test_data)

# Continue with the rest of your code...


sentiment_model = MultinomialNB()
sentiment_model.fit(train_features, train_labels)

# Make predictions on the training set
train_predictions = sentiment_model.predict(train_features)

# Evaluate the model on the training set
train_accuracy = accuracy_score(train_labels, train_predictions)
train_report = classification_report(train_labels, train_predictions)

# Make predictions on the test set
test_predictions = sentiment_model.predict(test_features)

# Evaluate the model on the test set
test_accuracy = accuracy_score(test_labels, test_predictions)
test_report = classification_report(test_labels, test_predictions)

# Print the results
print("Training Results:")
print(f"Accuracy: {train_accuracy:.2f}")
print("Classification Report:\n", train_report)

print("\nTesting Results:")
print(f"Accuracy: {test_accuracy:.2f}")
print("Classification Report:\n", test_report)


# # Saving Output 

# In[84]:


import pickle

with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(sentiment_model, model_file)


# In[ ]:





# In[ ]:




