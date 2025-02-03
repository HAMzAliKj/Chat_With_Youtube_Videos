from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import TFIDFRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key='AIzaSyC2rQzwSsl4a-sZnHWK_Kop7tlb53c3vRI')

# Streamlit app
st.title("YouTube Video Transcript Analyzer")

# Initialize session state variables
if "video_url" not in st.session_state:
    st.session_state.video_url = ''
if "last_url" not in st.session_state:
    st.session_state.last_url = ''
if "retriever" not in st.session_state:
    st.session_state.retriever = ''
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "subtitles" not in st.session_state:
    st.session_state.subtitles = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ''
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False     

# Input for YouTube video URL
video_url = st.text_input("Enter YouTube Video URL:")

# Check if a new URL is pasted
if video_url and video_url != st.session_state.last_url:
    st.session_state.video_url = video_url
    st.session_state.last_url = ''  # Reset last_url to allow reprocessing
    st.session_state.subtitles = None  # Reset subtitles
    st.session_state.transcript_text = ''  # Reset transcript text
    st.session_state.summary_generated = False  # Reset summary flag
    st.session_state.chat_history = []
 

# Fetch and process transcript
if st.session_state.video_url and not st.session_state.subtitles:
    video_id_prompt = f"Extract the YouTube video ID from this URL: {st.session_state.video_url}. Return only the video ID and nothing else."
    video_id = llm.invoke(video_id_prompt).content.strip()
    
    st.write(f"Extracted Video ID: `{str(video_id)}`")
    
    st.write("Select Video Language")
    language = st.selectbox("Select Video Language", options=("Hindi", "English", "Urdu"))
    
    if language:
        L = {"Hindi": "hi", "English": "en", "Urdu": "ur"}
        selected_language = L[language]
        
        with st.spinner(f"Waiting for 5 seconds before fetching {language} subtitles..."):
            time.sleep(5)  # Wait for 5 seconds
            
            st.session_state.subtitles = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_language])
            st.success(f"Language of Video Is {language}")

    try:
        if st.session_state.subtitles:
            # Store the transcript text in session state
            st.session_state.transcript_text = " ".join([subtitle['text'] for subtitle in st.session_state.subtitles])
            r_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
            page = r_splitter.split_text(st.session_state.transcript_text)
            st.session_state.retriever = TFIDFRetriever.from_texts(page)
            st.session_state.last_url = st.session_state.video_url  # Update last_url after processing
            st.success("Embedding Generated")

    except Exception as e:
        st.write(e)

# Summary button logic
if st.session_state.transcript_text and not st.session_state.summary_generated:
    summary = st.button("Generate Summary")
    if summary:
        analysis_prompt = f"Whats this video is mainly about summary in English: {st.session_state.transcript_text}"
        response = llm.invoke(analysis_prompt)                
        # Display the LLM's response
        st.write("LLM Response:")
        st.write(response.content)
        st.session_state.summary_generated = True  # Mark summary as generated


def response(user,chat_history):
    template = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful YouTube video assistant. Follow these rules:
         1. FIRST check chat history for answers to non-video questions
         2. Use video context ONLY when explicitly asked about content
         3. For rewrite/simplify requests, use previous answers from history
         4. Maintain natural conversation flow
         
         Current Chat History: {chat_history}
         Video Context: {context}"""),
         
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # inp = itemgetter("input")
    # con = itemgetter("chat_history")
    retriever_prompt = ChatPromptTemplate.from_messages([
         MessagesPlaceholder(variable_name="chat_history"),
         ("human","{input}"),
         ("human","Given the above conversation, generate the search query to look in order to get relevant information to the conversation")
    ])

    st.session_state.history_aware = create_history_aware_retriever(
         llm,
         retriever = st.session_state.retriever,
         prompt=retriever_prompt
         )

    # setup = {"input":inp,"chat_history":con,"context": inp | st.session_state.hisory_aware}
    
    chains = create_stuff_documents_chain(llm,template)
    result = create_retrieval_chain(
        st.session_state.history_aware,
        chains
        )
    answerb = result.invoke({"input":user,"chat_history":chat_history})
    return answerb['answer']
    # chain = setup | template | llm | StrOutputParser()
    # return chain.invoke({"input":user,"chat_history":chat_history})


try:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
                
    user = st.chat_input("How I Can Help You")

    if user: 
            st.session_state.chat_history.append(HumanMessage(user))

            with st.chat_message('user'):
                st.markdown(user)

            with st.chat_message('assistant'):
                try:
                    res = response(user,st.session_state.chat_history)
                    st.markdown(res)
                    st.session_state.chat_history.append(AIMessage(res))
                
                except Exception as e:
                        st.error(f"Error generating response: {e}")        


except Exception as e:
     st.write(e)