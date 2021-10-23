import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import base64
import querycat

DATA_URL = (
    "fuzzy-matching-template.csv"
)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

def checker(wrong_options,correct_options):
    names_array=[]
    ratio_array=[]
    for wrong_option in wrong_options:
        if wrong_option in correct_options:
            names_array.append(wrong_option)
            ratio_array.append('100')
        else:
            x=process.extractOne(wrong_option,correct_options,scorer=fuzz.token_set_ratio)
            names_array.append(x[0])
            ratio_array.append(x[1])
    return names_array,ratio_array

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="fuzzy-matching_template.csv">Download the template to populate</a>'
    return href

def get_table_download_link_two(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="matched_URLs.csv">Download csv file</a>'
    return href


with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
image = Image.open('seo-tools5.PNG')
st.sidebar.image(image)
st.text("")
st.sidebar.title("Available tools")
st.text("")
st.sidebar.markdown("### Which tool do you need?")
select = st.sidebar.selectbox('Choose tool', ['Fuzzy matching tool', 'Keyword categoriser'], key='1')
st.text("")

if select =='Fuzzy matching tool':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Fuzzy Matching Tool</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>This tool will give you the closest match for text (such as URLs), plus a score (out of 100) as to how close the match is.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Firstly, populate the following template:</strong></p>", unsafe_allow_html=True)
    st.markdown(get_table_download_link(data), unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>Populate <strong>Column1</strong> with the set of text or URLs that you want to lookup/match, and populate <strong>Column2</strong> with the text or URLs that you want to look up against.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Please only populate up to 5000 entries in each column otherwise it will break</strong>!</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>Now upload the populated file to get the matches:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv', key='3')
    if uploaded_file is not None:
        st.write("Matching...")
        df = pd.read_csv(uploaded_file, header=0)
        str2Match = df['Column1'].fillna('######').tolist()
        strOptions =df['Column2'].fillna('######').tolist()
        name_match,ratio_match=checker(str2Match,strOptions)
        df1 = pd.DataFrame()
        df1['Lookup_Text']=pd.Series(str2Match)
        df1['Matched_Text']=pd.Series(name_match)
        df1['Match_Ratio']=pd.Series(ratio_match)
        st.markdown('### Here are (up to) the first 50 matches')
        st.write("")
        matches = df1.head(50)
        st.write(matches)
        st.markdown('### Download the full dataset:')
        st.write("")
        st.markdown(get_table_download_link_two(matches), unsafe_allow_html=True)
if select =='Keyword categoriser':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Fuzzy Matching Tool</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>This tool will give you the closest match for text (such as URLs), plus a score (out of 100) as to how close the match is.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Firstly, populate the following template:</strong></p>", unsafe_allow_html=True)
    st.markdown(get_table_download_link(data), unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>Populate <strong>Column1</strong> with the set of text or URLs that you want to lookup/match, and populate <strong>Column2</strong> with the text or URLs that you want to look up against.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Please only populate up to 5000 entries in each column otherwise it will break</strong>!</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>Now upload the populated file to get the matches:</p>", unsafe_allow_html=True)
    keyword_file = st.file_uploader("Choose a CSV file", type='csv', key='4')
    if keyword_file is not None:
        st.write("Categorising...")
        dffinal = querycat.pd.read_csv(keyword_file, header=0)
        #catz = querycat.Categorize(dffinal, 'Keywords', min_support=2,  alg='apriori')
        #categories = catz.dffinal.head()
        #st.write(categories)
