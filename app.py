import streamlit as st
import wikipedia
import spacy_streamlit
import spacy
import requests

nlp = spacy.load('en_core_web_sm')

def main() :
	st.title("Streamlit App for NER")
	menu = ["NER - WIKI API", "NER - TEXT"]
	choice = st.sidebar.selectbox("Menu", menu)

	if(choice == "NER - WIKI API") :
		st.subheader("Named Entity Recognition using Wiki API")
		raw_text = st.text_area("Your text", "Enter text")
		# context = wikipedia.summary(raw_text)

		context = requests.get(f"https://flask-api-01.herokuapp.com/{raw_text}")
		context = context.json()
		print(context['data'])
		docx = nlp(context['data'])

		spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
	elif(choice == "NER - TEXT") :
		st.subheader("Named Entity Recognition using custom text")
		raw_text = st.text_area("Your text", "Enter text")
		docx = nlp(raw_text)
		spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)


if __name__ == '__main__':
	main()
