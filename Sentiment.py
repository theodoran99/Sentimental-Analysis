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
       # for file in upl:
           # if file.endswith('.csv'):
              #  df = pd.read_csv(file)
          #  elif file.endswith('.parquet'):
            #    df = pd.read_parquet(file)
            #    df.to_csv('parquet.csv', index = False)
          #  else:
           #     st.error("The file type is not supported")
 
        df = pd.read_csv(upl)
        column_name = st.selectbox('Select column for sentiment analysis:', df.columns)
        df['score'] = df[column_name].apply(score)
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
        lookup_words = st.text_input('Lookup words in CSV (separate by commas):')
        if lookup_words:
            lookup_words_list = [word.strip() for word in lookup_words.split(',')]
            for word in lookup_words_list:
                filtered_df = df[df[column_name].str.contains(word, case=False, na=False)]
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
                    plt.title(f'Sentiment Analysis Scatter Plot for "{lookup_words}"')
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

with st.expander('Compare CSVs'):
    file1 = st.file_uploader('Upload file 1')
    file2 = st.file_uploader('Upload file 2')

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

    if file1 and file2:
        data = []
        for file in [file1, file2]:
            file_name = file.name.lower() if file else None
            if file_name and file_name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file_name and file_name.endswith('.parquet'):
                df = pd.read_parquet(file)
                df.to_csv(f'{file_name}.csv', index=False)
            else:
                st.error("One or both of the file types are not supported")
                break  # Exit the loop if an unsupported file type is encountered
            data.append(df)
        if len(data) == 2:  # Ensure both files were processed successfully
            column_names = [st.selectbox(f'Select column for sentiment analysis (File {i}):', df.columns) for i, df in enumerate(data, start=1)]


            for i, df in enumerate(data, start=1):
                df['score'] = df[column_names[i-1]].apply(score)
                df['analysis'] = df['score'].apply(analyze)
                st.write(f"Data from File {i}:")
                st.write(df.head(10))

                # Create pie chart
                sentiment_counts = df['analysis'].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Create scatter plot
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df, x=df.index, y='score', hue='analysis', palette='viridis', ax=ax2)
                ax2.set_xlabel('Index')
                ax2.set_ylabel('Sentiment Score')
                ax2.set_title('Sentiment Analysis Scatter Plot')

                # Display both graphs side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Pie Chart:")
                    st.pyplot(fig1)
                with col2:
                    st.write("Scatter Plot:")
                    st.pyplot(fig2)

                positive_count = (df['analysis']== 'Positive').sum()
                neutral_count = (df['analysis']=='Neutral').sum()
                negative_count =(df['analysis']=='Negative').sum()
                # Download button
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                    label=f"Download data from File {i} as CSV",
                    data=csv,
                    file_name=f'sentiment_file{i}.csv',
                    mime='text/csv',
                    )
