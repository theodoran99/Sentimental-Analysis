from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Sentiment Analysis')

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_csv(upl)
        
        df['score'] = df['CONTENT'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        # Pie chart
        sentiment_counts = df['analysis'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.write(fig)


        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        # Scatter plot for sentiment analysis
        plt.figure()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.scatterplot(data=df, x=df.index, y='score', hue='analysis', palette='viridis')
        plt.xlabel('Index')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis Scatter Plot')
        st.pyplot()
        
                # Pop-up message
        positive_count = (df['analysis'] == 'Positive').sum()
        neutral_count = (df['analysis'] == 'Neutral').sum()
        negative_count = (df['analysis'] == 'Negative').sum()

        if positive_count > neutral_count and positive_count > negative_count:
            st.success("Thank you for sharing your CSV data. Our analysis reveals a predominantly positive sentiment, which is fantastic news. We'll continue to leverage this positivity to further enhance our offerings and ensure a consistently excellent user experience. Your input is invaluable, and we're excited to maintain this upward trajectory of positivity. ")
        elif neutral_count >= positive_count and neutral_count >= negative_count:
            st.info("Thank you for providing your CSV data. Our analysis indicates a neutral sentiment overall. While not leaning strongly in either direction, we view this as an opportunity to delve deeper into the data and identify areas for potential improvement. Your input is greatly appreciated, and we're committed to using it to enhance our services and better meet your needs.")
        else:
            st.error("Thank you for submitting your data. We've analyzed the sentiment from the provided CSV, and while the results lean towards the negative, we see this as an opportunity for growth and refinement. We'll delve into the data to identify key areas for improvement and work diligently to enhance the user experience. Your feedback is invaluable, and we're committed to striving for a more positive sentiment in the future.")

        # Lookup word in CSV and perform sentiment analysis
        lookup_word = st.text_input('Lookup word in CSV:')
        if lookup_word:
            filtered_df = df[df['CONTENT'].str.contains(lookup_word, case=False, na=False)]
            if not filtered_df.empty:
                sentiment_counts_lookup = filtered_df['analysis'].value_counts()
                fig_lookup, ax_lookup = plt.subplots()
                ax_lookup.pie(sentiment_counts_lookup, labels=sentiment_counts_lookup.index, autopct='%1.1f%%', startangle=90)
                ax_lookup.axis('equal')
                st.write(fig_lookup)

                # Scatter plot for the specific word
                plt.figure()
                sns.scatterplot(data=filtered_df, x=filtered_df.index, y='score', hue='analysis', palette='viridis')
                plt.xlabel('Index')
                plt.ylabel('Sentiment Score')
                plt.title(f'Sentiment Analysis Scatter Plot for "{lookup_word}"')
                st.pyplot()
            else:
                st.write("The word doesn't exist in the CSV content.")
            
            
             # Pop-up message
            positive_count2 = (df['analysis'] == 'Positive').sum()
            neutral_count2 = (df['analysis'] == 'Neutral').sum()
            negative_count2 = (df['analysis'] == 'Negative').sum()

            if positive_count2 > neutral_count2 and positive_count2 > negative_count2:
                st.success("Thank you for sharing your CSV data. Our analysis reveals a predominantly positive sentiment, which is fantastic news. We'll continue to leverage this positivity to further enhance our offerings and ensure a consistently excellent user experience. Your input is invaluable, and we're excited to maintain this upward trajectory of positivity. ")
            elif neutral_count2 >= positive_count2 and neutral_count2 >= negative_count2:
                st.info("Thank you for providing your CSV data. Our analysis indicates a neutral sentiment overall. While not leaning strongly in either direction, we view this as an opportunity to delve deeper into the data and identify areas for potential improvement. Your input is greatly appreciated, and we're committed to using it to enhance our services and better meet your needs.")
            else:
                st.error("Thank you for submitting your data. We've analyzed the sentiment from the provided CSV, and while the results lean towards the negative, we see this as an opportunity for growth and refinement. We'll delve into the data to identify key areas for improvement and work diligently to enhance the user experience. Your feedback is invaluable, and we're committed to striving for a more positive sentiment in the future.")


               
                
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
       

