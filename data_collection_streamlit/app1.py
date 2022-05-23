import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from time import sleep

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload




### App Set Up ###

st.set_page_config(layout="wide")
st.title('Codenames Dataset')
st.subheader('Instructions')
st.write("""
Read the list of words below. Can you think for a clue that links to them together and meets the following criteria?
- It must be a single word
- The clue cannot be one of the actual words
- The clue cannot be only rhyming

For example, a good clue for the words: Football, London and Stadium would be Arsenal. For Poison, Bow and Fire, a good clue would be Arrow.""")
st.write("")
st.write("""
Use the form below to select how many words you are linking together. More words is more difficult, but if you think you have a good clue, go for it!
Once you've selected your words, a box will appear for you to enter the clue. Make sure you press the 'submit your clue' button when you are happy.
""")
st.subheader('Your Words')

### Load words ###

@st.cache()
def select_words():
    words = pd.read_csv(Path(__file__).parents[0] / 'word_list.csv').sample(n=5).values
    words = [i[0] for i in words]
    return words

col1,col2,col3 = st.columns([1,1,2])
words = select_words()
for word in words[:3]:
    col1.write(word)
for word in words[3:]:
    col2.write(word)

### Skip Button ###
st.write("")
if st.button(label = 'Skip these words'):
    st.legacy_caching.clear_cache()
    st.experimental_rerun()
st.write("")

### Select words ###

selected_words = st.multiselect(label='Selected words:', options = words, default=None)

if selected_words:
    clue = st.text_input(label='Enter your single word clue:', value="")
    if clue:
        if st.button(label = 'Submit your clue'):
            datetimenow = datetime.now().strftime("%Y%m%d%H%M%S")
            results = {
                'words':';'.join(selected_words),
                'unselected_words': ';'.join([i for i in words if i not in selected_words]),
                'clue':clue}
            results = json.dumps(results)
            file_name = 'data_{}.json'.format(datetimenow)
            file_from = Path(__file__).parents[0] / file_name
            with open(file_from, 'w') as f:
                json.dump(results, f)

            #try:
            creds = Credentials(token=st.secrets['token'],
                                refresh_token = st.secrets['refresh_token'],
                                token_uri = st.secrets['token_uri'],
                                client_id=st.secrets['client_id'],
                                client_secret=st.secrets['client_secret'],
                                scopes=st.secrets['scopes'],
                                expiry=datetime.strptime(st.secrets['expiry'], "%Y-%m-%dT%H:%M:%S"))


            service = build('drive', 'v3', credentials=creds)
            file_metadata = {'name': file_name,
                             'parents': ['150dCgltvQa5gKum45F-fPlhKZ_tG0347']}
            media = MediaFileUpload(file_from)
            file = service.files().create(body=file_metadata,
                                          media_body=media,
                                          fields='id').execute()
            #except:
            #st.write(results)
            #st.error('Error: Contact Adam Shafi or Aaron Breuer-Weil')
            #sleep(10)

            st.legacy_caching.clear_cache()
            st.experimental_rerun()


