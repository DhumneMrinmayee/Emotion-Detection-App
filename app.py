

import streamlit as st 
import altair as alt
import plotly.express as px 


# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 

#function

lr_pipe = joblib.load(open("models/Sentiment_classifier_LR_Pipeline_Jan_2023.pkl", "rb"))


# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table


def predict_sentiment(docx):
	results = lr_pipe.predict([docx])
	return results[0]

def get_predict_proba(docx):
	results = lr_pipe.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😡😠","boredom":"🤮", "empty":"😨😱", "love":"🤗🥰", "fun":"😂", "neutral":"😐","sadness":"😔", "worry":"😳", "surprise":"😮", "happiness":"😊", "enthusiasm":"🤩", "hate":"🤬", "relife":"😮‍💨"}

#main Application

def main():
	st.title("Sentiment Analyzer")
	menu = ['Home', 'Dashboard','About']
	choice = st.sidebar.selectbox("Menu", menu)
	create_page_visited_table()
	create_emotionclf_table()

	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.subheader("Home \n Sentiments In Text")

		with st.form(key = "Sentiment clf form"):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label= "Submit")

		if submit_text:
			col1, col2 = st.columns(2)

			#apply function
			prediction = predict_sentiment(raw_text)
			probability = get_predict_proba(raw_text)

			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{} {}".format(prediction, emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))


			with col2:
				st.success("Prediction Probability")
				#st.write(probability)
				proba_df = pd.DataFrame(probability, columns = lr_pipe.classes_)
				#st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["Sentiments", "Probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x= "Sentiments", y = "Probability", color='Sentiments')
				st.altair_chart(fig, use_container_width=True)


	elif choice == "Dashboard":
		add_page_visited_details("Dashboard",datetime.now())
		st.subheader("Dashboard App")


		
		with st.expander('Sentiment Classifier Metrics'):
			df_sentiment = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_sentiment)

			prediction_count = df_sentiment['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)

		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart(c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)
	



	else:
		st.subheader("About \n This is an NLP powered webapp that can predict emotion from text recognition with 70 percent accuracy")
		add_page_visited_details("About",datetime.now())



if __name__ == '__main__':
	main()