import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import base64
import querycat
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from prophet import Prophet
from pytrends.request import TrendReq
import requests
import random
import csv
import datetime
from datetime import date
from datetime import datetime
import time
import argparse
#pytrends = TrendReq(proxies=[st.secrets["PROXY"]], retries=5, backoff_factor=5, requests_args={'verify':False})
pytrends = TrendReq()
import fasttext
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords
import re
import string

import json

DATA_URL = (
    "fuzzy-matching-template.csv"
)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

DATA_URL2 = (
    "keyword_categoriser_template.csv"
)

@st.cache(persist=True)
def load_data():
    data2 = pd.read_csv(DATA_URL2)
    return data2

data2 = load_data()

DATA_URL3 = (
    "forecast-template.csv"
)

@st.cache(persist=True)
def load_data():
    data3 = pd.read_csv(DATA_URL3)
    return data3

data3 = load_data()

COVID_lockdown1 = pd.DataFrame({
    'holiday': 'covid',
    'ds':  pd.date_range(start='2020-03-23',
                         end='2020-06-23',
                         freq='D'),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 1
    })

COVID_lockdown2 = pd.DataFrame({
    'holiday': 'covid',
    'ds':  pd.date_range(start='2020-11-05',
                         end='2020-12-02',
                         freq='D'),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 1
    })

COVID_lockdown3 = pd.DataFrame({
    'holiday': 'covid',
    'ds':  pd.date_range(start='2021-01-06',
                         end='2021-07-19',
                         freq='D'),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 1
    })

holidays = pd.concat((COVID_lockdown1, COVID_lockdown2, COVID_lockdown3))


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
    href = f'<a href="data:file/csv;base64,{b64}" download="fuzzy_matching_template.csv">Download the template to populate</a>'
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

def get_table_download_link_three(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="keyword_categoriser_template.csv">Download the template to populate</a>'
    return href

def get_table_download_link_four(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="categorised_keywords.csv">Download csv file</a>'
    return href

def get_table_download_link_five(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="top-performers.csv">Download all top performer data</a>'
    return href

def get_table_download_link_six(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast-template.csv">Download forecast template</a>'
    return href

def get_table_download_link_seven(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast-data.csv">Download forecast data</a>'
    return href

def get_table_download_link_eight(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="classified-queries.csv">Download classified queries</a>'
    return href

def get_table_download_link_nine(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="trend-data.csv">Download trend data</a>'
    return href

def get_table_download_link_ten(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="trend-data.csv">Download volume data</a>'
    return href

# Cache trained model
#@st.experimental_singleton
def get_model():
    model = fasttext.train_supervised(input="train.txt")
    return model

# Save trained model to file
def save_model(model, path="model.bin"):
    model.save_model(path)

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
image = Image.open('seo-tools5.PNG')
st.sidebar.image(image)
st.text("")
st.sidebar.title("Available tools")
st.text("")
st.sidebar.markdown("### Which tool do you need?")
select = st.sidebar.selectbox('Choose tool', ['Bulk Google Trends tool', 'Search volume gatherer', 'Forecasting tool', 'Fuzzy matching tool', 'Keyword categoriser', 'Text classifier', 'SERP top performer analysis'], key='1')
st.text("")
if select =='Bulk Google Trends tool':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Bulk Google Trends Tool</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>1. Make a list of phrases you want to check in Google Sheets. <strong>Please stick to 20 phrases each day</strong>.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>2. Download the sheet as a <strong>.csv file</strong>.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>3. Choose to get trends from the <strong>UK, US or Worldwide</strong>:</p>", unsafe_allow_html=True)
    country_input = st.selectbox('Where do you want to get trends from?', ['UK', 'US', 'Worldwide'], key='12')
    if country_input == 'UK':
        country_input = 'GB'
    if country_input == 'Worldwide':
        country_input = ''
    st.markdown("<p style='font-weight:normal'>4. <strong>Drop the file here:</strong></p>", unsafe_allow_html=True)
    trends_file = st.file_uploader("Choose a CSV file", type='csv', key='8')
    if trends_file is not None:
        st.write("Getting trends data...")
        f = pd.read_csv(trends_file, header=None)
        finaldff = pd.DataFrame()
        for index, row in f.iterrows():
                kw_list = row
                name = kw_list[0]
                try:
                    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo=country_input, gprop='')
                    df = pytrends.interest_over_time()
                    time.sleep(5  + random.random())
                    try:
                        finaldff[name] = df[name]
                    except KeyError:
                        finaldff[name] = 'skipped'
                except requests.exceptions.Timeout:
                    st.write("Timeout occured")
                except requests.exceptions.ConnectionError:
                    st.write("Connection error")
        testdf = finaldff[256:].replace(0, 'skipped')
        finaldff2 = finaldff
        finaldff2 = finaldff2.append(testdf)
        finaldff2 = finaldff2.loc[:,(finaldff2!='skipped').all()]
        #finaldff2 = finaldff2.drop(['Query'], axis=1)
        finaldff3 = finaldff2[207:260]
        finaldff3 = finaldff3.transpose()
        columns = len(finaldff3.columns)
        last_column = columns - 1
        second_last_column = columns - 2
        third_last_column = columns - 3
        fourth_last_column = columns - 4
        fifth_last_column = columns - 5
        sixth_last_column = columns - 6
        seventh_last_column = columns - 7
        eighth_last_column = columns - 8
        ninth_last_column = columns - 9
        tenth_last_column = columns - 10
        finaldff3['last_threeweekavg'] = (finaldff3.iloc[:, last_column] + finaldff3.iloc[:, second_last_column] + finaldff3.iloc[:, third_last_column])/3
        finaldff3['first_threeweekavg'] = (finaldff3.iloc[:, 0] + finaldff3.iloc[:, 1] + finaldff3.iloc[:, 2]+ + finaldff3.iloc[:, 3] + + finaldff3.iloc[:, 4])/5
        finaldff3['score'] = finaldff3['last_threeweekavg'] - finaldff3['first_threeweekavg'] + finaldff3.iloc[:, last_column]
        finaldff3 = finaldff3.sort_values(by=['score'], ascending=False)
        finaldff4 = finaldff3.drop(['last_threeweekavg'], axis=1)
        finaldff4 = finaldff4.drop(['first_threeweekavg'], axis=1)
        finaldff4 = finaldff4.drop(['score'], axis=1)
        tabletosend = finaldff4
        tabletosend['Week on Week % Change'] = (((finaldff3.iloc[:, last_column]/finaldff3.iloc[:, second_last_column])*100)-100).round(0).astype(str) + '%'
        tabletosend['Month on Month % Change'] = (((finaldff3.iloc[:, last_column]/finaldff3.iloc[:, fifth_last_column])*100)-100).round(0).astype(str) + '%'
        tabletosend['Year on Year % Change'] =(((finaldff3.iloc[:, last_column]/finaldff3.iloc[:, 0])*100)-100).round(0).astype(str) + '%'
        tabletosend = tabletosend.transpose()
        tabletosend = tabletosend.tail(3)
        tabletosend = tabletosend.transpose()
        st.write(tabletosend)
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_nine(tabletosend), unsafe_allow_html=True)
        finaldff3 = finaldff3.transpose()
        counter = 1
        for column in finaldff3:
            fig = go.Figure()
            fig.update_layout(
                title={
                'text': column,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
                xaxis_title="Week",
                yaxis_title="Google Trend Score",
                xaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                #ticks='outside',
                tickfont=dict(
                family='Interval',
                size=14,
                #color='rgb(82, 82, 82)',
                ),
                ),
                yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=True,
                ),
                showlegend=False,
                #plot_bgcolor='white'
                )
            fig.add_trace(go.Scatter(x=finaldff3.index, y=finaldff3[column], name=column, line=dict(color='hotpink', width=4)))
            counter = counter + 1
            st.plotly_chart(fig)
if select =='Search volume gatherer':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Search volume gatherer</strong></h2>", unsafe_allow_html=True)
    st.markdown('### 1. Which areas do you want to get volume in?')
    select = st.selectbox('Choose location', ['Top 10 UK cities', 'All 50 US states', 'My own list of locations'], key='13')
    if select =='My own list of locations':
        st.markdown("<p style='font-weight:normal'><strong>Upload your list of locations</strong></p>", unsafe_allow_html=True)
        locations_file = st.file_uploader("Choose a CSV file", type='csv', key='15')
    st.markdown("### 2. Upload your list of queries with the column heading of 'keywords'")
    volumes_file = st.file_uploader("Choose a CSV file", type='csv', key='14')
    if volumes_file is not None:
        volumes_file = pd.read_csv(volumes_file)
        df_list = volumes_file['keywords'].tolist()
        if locations_file is not None:
            if select =='All 50 US states':
                locations_file = pd.read_csv('US-States.csv', header=None)
            if select == 'Top 10 UK cities':
                locations_file = pd.read_csv('UK-Cities.csv', header=None)
            if select == 'My own list of locations':
                locations_file = pd.read_csv(locations_file, header=None)
            finalfinal_frame = pd.DataFrame(columns=['keyword', 'volume', 'Location ID'])
            for index, row in locations_file.iterrows():
                location = row
                location = int(location)
                url = "https://api.keywordtool.io/v2/search/volume/google"
                data = {
                "apikey": st.secrets["KEYWORDTOOL"],
                "keyword":
                df_list,
                "metrics_location": [
                location
                ],
                "metrics_language": [
                "en"
                ],
                "metrics_network": "googlesearchnetwork",
                "metrics_currency": "GBP",
                "output": "json"
                }
                response = requests.post(url, json = data)
                #print(response.text.encode('utf8'))
                def jprint(obj):
                    # create a formatted string of the Python JSON object
                    text = json.dumps(obj, sort_keys=True, indent=4)
                    #data = jprint(response.json())
                data = response.json()
                json_str = json.dumps(data)
                resp = json.loads(json_str)
                final_frame = pd.DataFrame(columns=['keyword', 'volume', 'Location ID', 'Month 1', 'Month 2','Month 3','Month 4','Month 5','Month 6','Month 7','Month 8','Month 9','Month 10','Month 11','Month 12'])
                for i in df_list:
                    keyword = i
                    volume = resp['results'][keyword]['volume']
                    month1 = resp['results'][keyword]['m1']
                    month2 = resp['results'][keyword]['m2']
                    month3 = resp['results'][keyword]['m3']
                    month4 = resp['results'][keyword]['m4']
                    month5 = resp['results'][keyword]['m5']
                    month6 = resp['results'][keyword]['m6']
                    month7 = resp['results'][keyword]['m7']
                    month8 = resp['results'][keyword]['m8']
                    month9 = resp['results'][keyword]['m9']
                    month10 = resp['results'][keyword]['m10']
                    month11 = resp['results'][keyword]['m11']
                    month12 = resp['results'][keyword]['m12']
                    table = {'keyword':keyword,'volume':[volume],'Location ID':[location], 'Month 1': month1, 'Month 2': month2, 'Month 3': month3, 'Month 4': month4, 'Month 5': month5, 'Month 6': month6, 'Month 7': month7, 'Month 8': month8, 'Month 9': month9, 'Month 10': month10, 'Month 11': month11, 'Month 12': month12}
                    dataframe = pd.DataFrame(table)
                    final_frame = final_frame.append(dataframe)
                finalfinal_frame = finalfinal_frame.append(final_frame)
                time.sleep(7  + random.random())
        df_reference = pd.read_csv('all_locations_keywordtool.csv')
        joined_df = pd.merge(finalfinal_frame,
                 df_reference,
                 on ='Location ID',
                 how ='inner')
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_ten(joined_df), unsafe_allow_html=True)
if select =='Forecasting tool':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Forecasting tool</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Firstly, populate the following template, keeping the column headings exactly as they are:</strong></p>", unsafe_allow_html=True)
    st.markdown(get_table_download_link_six(data3), unsafe_allow_html=True)
    frequency_input = st.selectbox('What is the frequency of your data?', ['Daily', 'Weekly', 'Monthly'], key='8')
    if frequency_input == 'Daily':
        frequency_input = 'D'
    if frequency_input == 'Weekly':
        frequency_input = 'W'
    if frequency_input == 'Monthly':
        frequency_input = 'M'
    period_input = st.text_input('How many periods into the future to do you want to forecast?', 365)
    period_input = int(period_input)
    national_holidays = st.selectbox('Which national holidays do you want to add?', ['UK', 'US', 'Neither'], key='19')
    lockdowns = st.selectbox('Do you want to add UK COVID-19 lockdowns?', ['Yes', 'No'], key='20')
    st.markdown("<p style='font-weight:normal'><strong>Now upload the populated file to get the forecast:</strong></p>", unsafe_allow_html=True)
    forecast_file = st.file_uploader("Choose a CSV file", type='csv', key='7')
    if forecast_file is not None:
        st.write("Forecasting...")
        df = pd.read_csv(forecast_file)
        df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
        if lockdowns == 'Yes':
            m = Prophet(holidays=holidays)
        else:
            m = Prophet()
        if national_holidays != 'Neither':
            m.add_country_holidays(country_name=national_holidays)
        m.fit(df)
        future = m.make_future_dataframe(periods=period_input, freq=frequency_input)
        future.tail()
        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        dffinal = pd.DataFrame()
        dffinal['date'] = forecast['ds']
        dffinal['actual'] = df['y']
        dffinal['forecast'] = forecast['yhat']
        dffinal = dffinal.fillna('')
        dffinal = dffinal.set_index('date')
        fig = go.Figure()
        fig.add_trace(go.Line(x=dffinal.index, y=dffinal['actual'], name='Actual',
                         text=dffinal['actual'], line=dict(color='#ff0bac', width=3)
                        ))
        fig.add_trace(go.Line(x=dffinal.index, y=dffinal['forecast'], name='Forecast',
                        text=dffinal['forecast'], line=dict(color='#a13bff', width=3)
                        ))
        fig.update_layout(
            plot_bgcolor="#ffffff",
            title='Forecast',
            titlefont_size=20,
            title_x=0.5,
            xaxis=dict(
                title='Date',
                titlefont_size=16,
                tickfont_size=14,
            ),
            yaxis=dict(
                title='Metric',
                titlefont_size=16,
                tickfont_size=14,
            ),
            legend=dict(
                xanchor = "center",
                x=0.5,
                yanchor="bottom",
                y=1.02,
                orientation="h",
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1 # gap between bars of the same location coordinate.
        )
        st.plotly_chart(fig)
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_seven(dffinal), unsafe_allow_html=True)
if select =='Fuzzy matching tool':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Fuzzy Matching Tool</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>This tool will give you the closest matches between two columns of text (such as URLs), plus a score (out of 100) as to how close the match is.</p>", unsafe_allow_html=True)
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
        st.markdown(get_table_download_link_two(df1), unsafe_allow_html=True)
if select =='Keyword categoriser':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Keyword Categoriser</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>This tool uses <strong>counts of keywords</strong> to quickly find common themes from a list of search queries. It uses <a href='https://github.com/jroakes/querycat'>Querycat from JR Oakes.</a></p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Firstly, populate the following template - putting your list into the 'Keywords' column:</strong></p>", unsafe_allow_html=True)
    st.markdown(get_table_download_link_three(data2), unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'><strong>Then choose how many times you want a keyword to be mentioned before it becomes a category (test this with multiple options to find what works best).</strong></p>", unsafe_allow_html=True)
    user_input = st.text_input("How many mentions for a keyword before you want to count it as a category?", 3)
    user_input = int(user_input)
    st.markdown("<p style='font-weight:normal'><strong>Now upload the populated template:</strong></p>", unsafe_allow_html=True)
    keyword_file = st.file_uploader("Choose a CSV file", type='csv', key='4')
    if keyword_file is not None:
        st.write("Categorising...")
        df = querycat.pd.read_csv(keyword_file)
        catz = querycat.Categorize(df, 'Keywords', min_support=user_input,  alg='apriori')
        catz.df.drop('match_queries', axis=1, inplace=True)
        catz.df.rename(columns={'category': 'Category'}, inplace=True)
        #st.markdown('### Here are the top 50 categories and the count of queries included in each:')
        #st.write("")
        #categories = catz.counts.head(50)
        #st.write(categories)
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_four(catz.df), unsafe_allow_html=True)
if select =='Text classifier':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Text Classifier</strong></h2>", unsafe_allow_html=True)
    st.markdown('### 1. Do you already have a trained model?')
    nope = st.checkbox("No")
    yesss = st.checkbox("Yes")
    if nope:
        st.markdown('### Train your model:')
        st.markdown("<p style='font-weight:normal'>Upload a file with the column headings <strong>'Keywords'</strong> and <strong>'Category'</strong>.</p>", unsafe_allow_html=True)
        epoch_input = st.text_input("How many epochs do you want to train with?", 100)
        epoch_input = int(epoch_input)
        ngram_input = st.text_input("How many ngrams do you want to use?", 2)
        ngram_input = int(ngram_input)
        dataset = st.file_uploader("Choose a CSV file", type='csv', key='8')
        if dataset is not None:
            # NLP Preprocess
            st.write("Training...")
            dataset = pd.read_csv(dataset)
            stop = stopwords.words('english')
            #Apply the removal of the stopwords to the newly added cleaned reviews column
            dataset['Keywords'] = dataset['Keywords'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            dataset['Keywords'] = dataset['Keywords'].apply(lambda x: ' '.join(simple_preprocess(x)))
            # Prefixing each row of the category column with '__label__'
            dataset['Category'] = dataset['Category'].apply(lambda x: '__label__' + x)
            dataset[['Category', 'Keywords']].to_csv('train.txt',
                                          index = False,
                                          sep = ' ',
                                          header = None,
                                          quoting = csv.QUOTE_NONE,
                                          quotechar = "",
                                          escapechar = " ")
            dataset[['Category', 'Keywords']].to_csv('test.txt',
                                          index = False,
                                          sep = ' ',
                                          header = None,
                                          quoting = csv.QUOTE_NONE,
                                          quotechar = "",
                                          escapechar = " ")
            model = fasttext.train_supervised('train.txt', epoch=epoch_input, wordNgrams = ngram_input)
            st.write("Model is trained")
            test_results = model.test('test.txt')
            st.write("The test results are:")
            st.write(test_results)
            model = get_model()
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            file_time = "model -" + str(timestamp) + ".bin"
            save_model(model, path=file_time)
            # Download saved trained model
            with open(file_time, "rb") as f:
                btn = st.download_button(
                    label="Download trained text classification model",
                    data=f,
                    file_name=file_time # Any file name
                 )
    if yesss:
        st.markdown('### Upload your model:')
        uploaded_model = st.file_uploader("Choose a model file", type='bin', key='12')
        if uploaded_model is not None:
            import os
            def save_uploadedfile(uploadedfile):
                with open(uploaded_model.name,"wb") as f:
                    f.write(uploadedfile.getbuffer())
                return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
            file_details = {"FileName":uploaded_model.name,"FileType":uploaded_model.type}
            save_uploadedfile(uploaded_model)
            model = fasttext.load_model(uploaded_model.name)
            stop = stopwords.words('english')
    st.markdown('### 2. Classify your queries:')
    st.markdown("<p style='font-weight:normal'>Upload a file with the column heading <strong>'Keywords'</strong>.</p>", unsafe_allow_html=True)
    classify = st.file_uploader("Choose a CSV file", type='csv', key='9')
    if classify is not None:
        st.write("Classifying...")
        model = model
        classify = pd.read_csv(classify)
        #stop = stopwords.words('english')
        #Apply the removal of the stopwords to the newly added cleaned reviews column
        classify['Processed'] = classify['Keywords'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        classify['Processed']  = classify['Processed'].apply(lambda x: ' '.join(simple_preprocess(x)))
        def prediction(category):
            '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
            predict = model.predict(category)
            return predict
        round1 = lambda x: prediction(x)
        #Apply the function and create a new column
        classify['Predictions'] = classify.Processed.apply(round1)
        classify['Predictions'] = classify['Predictions'].astype(str)
        classify['Predictions'] = classify['Predictions'].replace(regex={r'\(\(\'__label__': ''})
        classify['Predictions'] = classify['Predictions'].replace(regex={r'\'.*': ''})
        classify.drop(['Processed'], axis=1, inplace=True)
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_eight(classify), unsafe_allow_html=True)
if select =='SERP top performer analysis':
    st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>SERP Top Performer Analysis</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>This tool converts an <strong>SEOmonitor report</strong> into a graph showing the <strong>top performers</strong> for a keyword set.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:normal'>1. Choose if you want <strong>estimated organic traffic</strong>, <strong>visibility score</strong> or <strong>both (tick both boxes)</strong>:</p>", unsafe_allow_html=True)
    traffic = st.checkbox("Estimated organic search traffic")
    visibility = st.checkbox("Visibility score")
    st.markdown("<p style='font-weight:normal'>2. Upload the <strong>SERPs Data</strong> report from <strong>SEOMonitor</strong> here:</p>", unsafe_allow_html=True)
    visibility_file = st.file_uploader("Choose a CSV file", type='csv', key='5')
    if visibility_file is not None:
        st.write("Finding top performers...")
        df = pd.read_csv(visibility_file)
        df2 = pd.read_csv('visibility-scores.csv')
        df3 = pd.read_csv('click-through-rates.csv')
        df['Domain'] = df['Landing page 1']
        df['Domain'] = df['Domain'].replace(regex={r'https:\/\/': ''})
        df['Domain'] = df['Domain'].replace(regex={r'http:\/\/': ''})
        df['Domain'] = df['Domain'].replace(regex={r'www\.': ''})
        df['Domain'] = df['Domain'].replace(regex={r'/.*': ''})
        df.sort_values(by=['Rank'], axis=0, ascending=True, inplace=True)
        df['Dedupe Column'] = df['Keyword'] + df['Device'] + df['Domain']
        df.drop_duplicates(subset=['Dedupe Column'], inplace=True)
        df.drop('Dedupe Column', axis=1, inplace=True)
        inner_join = pd.merge(df,
                  df2,
                  on ='Rank',
                  how ='inner')
        inner_join2 = pd.merge(inner_join,
                  df3,
                  on ='Rank',
                  how ='inner')
        inner_join2['Search Volume'] = inner_join2['Search Volume'].astype(int)
        inner_join2['CTR'] = inner_join2['CTR'].astype(float)
        inner_join2['Estimated_Traffic_Score'] = inner_join2['Search Volume'] * inner_join2['CTR']
        score_table = inner_join2.filter(items=['Device', 'Domain', 'Visibility_Score', 'Estimated_Traffic_Score'])
        value_list = ["M"]
        boolean_series = score_table.Device.isin(value_list)
        visibility_score_mobile = score_table[boolean_series]
        visibility_score_mobile = visibility_score_mobile.groupby('Domain')
        visibility_score_sum = visibility_score_mobile.sum()
        visibility_score_sum.sort_values(by=['Estimated_Traffic_Score'], axis=0, ascending=False, inplace=True)
        keyword_counts = inner_join2.filter(items=['Keyword', 'Search Volume'])
        keyword_counts.drop_duplicates(subset=['Keyword'], inplace=True)
        total_possible_visibility = len(keyword_counts) * 20
        total_possible_traffic = keyword_counts['Search Volume'].sum() * 0.1507
        visibility_score_sum['Visibility_Score_Percentage'] = round((visibility_score_sum['Visibility_Score'] / total_possible_visibility) * 100)
        visibility_score_sum['Traffic_Score_Percentage'] = round((visibility_score_sum['Estimated_Traffic_Score'] / total_possible_traffic) * 100)
        visibility_score_sum['Visibility Score Percentage'] = visibility_score_sum['Visibility_Score_Percentage'].astype(int)
        visibility_score_sum['Traffic Score Percentage'] = visibility_score_sum['Traffic_Score_Percentage'].astype(int)
        final_df = visibility_score_sum.head(5)
        fig = go.Figure()
        if traffic:
            fig.add_trace(go.Bar(x=final_df.index, y=final_df['Traffic Score Percentage'], name='Estimated Traffic Score',
                        marker_color='#ff0bac', text=final_df['Traffic Score Percentage']
                        ))
        if not traffic:
            visibility_score_sum.sort_values(by=['Visibility Score Percentage'], axis=0, ascending=False, inplace=True)
            final_df = visibility_score_sum.head(5)
        if visibility:
            fig.add_trace(go.Bar(x=final_df.index, y=final_df['Visibility Score Percentage'], name='Visibility Score',
                        marker_color='#a13bff', text=final_df['Visibility Score Percentage']
                        ))
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_layout(
            plot_bgcolor="#ffffff",
            title='Top 5 Industry Performers',
            titlefont_size=20,
            title_x=0.5,
            xaxis_tickfont_size=14,
            yaxis=dict(
                title='Percentage (of Best Possible)',
                titlefont_size=16,
                tickfont_size=14,
            ),
            legend=dict(
                xanchor = "center",
                x=0.5,
                yanchor="bottom",
                y=1.05,
                orientation="h",
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1 # gap between bars of the same location coordinate.
        )
        st.plotly_chart(fig)
        visibility_score_sum.drop('Visibility_Score', axis=1, inplace=True)
        visibility_score_sum.drop('Estimated_Traffic_Score', axis=1, inplace=True)
        visibility_score_sum.drop('Visibility_Score_Percentage', axis=1, inplace=True)
        visibility_score_sum.drop('Traffic_Score_Percentage', axis=1, inplace=True)
        st.markdown('### Download the full dataset:')
        st.markdown(get_table_download_link_five(visibility_score_sum), unsafe_allow_html=True)
