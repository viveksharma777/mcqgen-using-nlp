import streamlit as st
from streamlit_lottie import st_lottie
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
import requests
import random
import re
import PyPDF2
from PyPDF2 import PdfReader

from rake_nltk import Rake

r = Rake()



# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('popular')

# Define the function to extract important words
def getImportantWords(art):
    r = Rake(stopwords=stopwords.words('english') + list(string.punctuation))
    r.extract_keywords_from_text(art)
    keyphrases = r.get_ranked_phrases_with_scores()
    result = [keyphrase for _, keyphrase in keyphrases[:25]]  # Get the top 25 keyphrases
    return result

# Split the text into sentences
def splitTextToSents(art):
    s = [sent_tokenize(art)]
    s = [y for x in s for y in x]
    s = [sent.strip() for sent in s if len(sent) > 15]
    return s

# Map sentences to keywords
def mapSents(impWords, sents):
    processor = KeywordProcessor()
    keySents = {}
    for word in impWords:
        keySents[word] = []
        processor.add_keyword(word)
    for sent in sents:
        found = processor.extract_keywords(sent)
        for each in found:
            keySents[each].append(sent)
    for key in keySents.keys():
        temp = keySents[key]
        temp = sorted(temp, key=len, reverse=True)
        keySents[key] = temp
    return keySents

# Get the sense of the word
def getWordSense(sent, word):
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Get distractors from WordNet
def getDistractors(syn, word):
    dists = []
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return dists
    for each in hypernym[0].hyponyms():
        name = each.lemmas()[0].name()
        if name == actword:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in dists:
            dists.append(name)
    return dists

# Get distractors from ConceptNet
def getDistractors2(word):
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    dists = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in dists and actword.lower() not in word2.lower():
                dists.append(word2)
    return dists


# Load Lottie animation from a URL
lottie_animation = load_lottie_url("https://lottie.host/a17cef35-9c09-474f-9c26-225130dab967/dWJO2lpTSY.json")

    
# Streamlit App
st.markdown("<h1 style='color: skyblue;'>MCQ Generator</h1>", unsafe_allow_html=True)


# Description
st.markdown("""
## Generate Multiple Choice Questions from Text
This app allows you to generate multiple choice questions (MCQs) from any text you provide. 
Simply upload a text file, specify the number of questions, and click the "Generate MCQs" button.
""")


# Align the info button to the top right corner
col1, col2 = st.columns([10, 2])
with col1:
    pass  # Empty column to adjust the layout

with col2:
    # Info button
    if st.button(label= "ℹ️ Info", key='info_button', help="Click for info"):
        st.sidebar.title("Information")
        st.sidebar.markdown("""
        Our project, the MCQ Generator, automatically generates multiple-choice questions (MCQs) from uploaded text files. 
        It utilizes natural language processing techniques to extract important words, map them to sentences, and provide 
        distractors for each question, enhancing learning and assessment processes in various educational contexts. 

        **Team Members:**
        -  Biswajit Kar
        -  Vivek Sharma
        -  Bishal Sarmah

        **References:**
        - [MCQ Generator Jupyter Notebook](https://github.com/vaishnaviverma/MCQ-Generator/blob/main/MCQG.ipynb)
        - [Lottie Files](https://lottiefiles.com)
        """)


    
# Display Lottie animation
if lottie_animation:
    st_lottie(lottie_animation, height=300, key="coding")
else:
    st.error("Error loading Lottie animation. Please check the URL or try again later.")
    
        

## File upload handler for text files
uploaded_file = st.file_uploader("Choose a text file", type="txt")
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    
    num_mcqs = st.number_input("Enter the number of questions you want:", min_value=1, value=5)

    if st.button('Generate MCQs', key='generate_button'):
        impWords = getImportantWords(text)
        sents = splitTextToSents(text)
        mappedSents = mapSents(impWords, sents)

        mappedDists = {}
        correctAnswers = {}

        for each in impWords:
            if each not in mappedSents or not mappedSents[each]:
                continue
            distractors = random.sample([k for k in impWords if k != each], 3)
            mappedDists[each] = distractors

        
        iterator = 1  # To keep the count of the questions
        for each in mappedDists:
            if iterator > num_mcqs:
                break  # exit the loop if the desired number of MCQs has been reached
            if each not in mappedSents or not mappedSents[each]:  # Check if the keyword is not in mappedSents or if its list of sentences is empty
                continue  # Skip this keyword if it's not found in mappedSents or has no mapped sentences
            sent = mappedSents[each][0]
            p = re.compile(each, re.IGNORECASE)  # Converts into regular expression for pattern matching
            op = p.sub("________", sent)  # Replaces the keyword with underscores(blanks)
            correct_answer = each.capitalize()  # The correct answer
            st.write(f"**Question {iterator}**: {op}")  # Prints the question along with a question number
            options = [each.capitalize()] + mappedDists[each]  # Capitalizes the options
            options = options[:4]  # Selects only 4 options
            opts = ['a', 'b', 'c', 'd']
            random.shuffle(options)  # Shuffle the options so that order is not always same
            for i, ch in enumerate(options):
                st.write(f"\t {opts[i]}) {ch}") # Print the options
            st.write(f"**Correct Answer**: {correct_answer}")  # Print the correct answer
            st.write("\n")
            iterator += 1  # Increase the counter
    