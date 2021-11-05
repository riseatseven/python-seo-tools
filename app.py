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

if 'password' not in st.session_state:
    password = st.text_input('Enter password', value='', type='password')
    if not password=='R1s3Up!':
        st.session_state.password = False

else:

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
        "forecasting-template.csv"
    )

    @st.cache(persist=True)
    def load_data():
        data3 = pd.read_csv(DATA_URL2)
        return data3

    data3 = load_data()


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

    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    image = Image.open('seo-tools5.PNG')
    st.sidebar.image(image)
    st.text("")
    st.sidebar.title("Available tools")
    st.text("")
    st.sidebar.markdown("### Which tool do you need?")
    select = st.sidebar.selectbox('Choose tool', ['Forecasting tool', 'Fuzzy matching tool', 'Keyword categoriser', 'SERP top performer analysis'], key='1')
    st.text("")
    if select =='Forecasting tool':
        st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Forecasting tool</strong></h2>", unsafe_allow_html=True)
        forecast_file = st.file_uploader("Choose a CSV file", type='csv', key='7')
        st.markdown("<p style='font-weight:normal'><strong>Firstly, populate the following template:</strong></p>", unsafe_allow_html=True)
        st.markdown(get_table_download_link(data3), unsafe_allow_html=True)
        if forecast_file is not None:
            st.write("Forecasting...")
            df = pd.read_csv(forecast_file)
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=365)
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
            st.markdown(get_table_download_link_seven(dffinal), unsafe_allow_html=True
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
            st.markdown(get_table_download_link_two(matches), unsafe_allow_html=True)
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
            st.markdown('### Here are (up to) the first 50 keywords and the category for each:')
            st.write("")
            categories = catz.df.head(50)
            st.write(categories)
            st.markdown('### Download the full dataset:')
            st.markdown(get_table_download_link_four(catz.df), unsafe_allow_html=True)
    if select =='SERP top performer analysis':
        st.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>SERP Top Performer Analysis</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>Upload the <strong>SERPs Data</strong> report from <strong>SEOMonitor</strong> here:</p>", unsafe_allow_html=True)
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
            fig.add_trace(go.Bar(x=final_df.index, y=final_df['Traffic Score Percentage'], name='Estimated Traffic Score',
                            marker_color='#ff0bac', text=final_df['Traffic Score Percentage']
                            ))
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
