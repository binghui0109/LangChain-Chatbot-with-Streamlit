import os
import sys
import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import time
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from PIL import Image 
from langchain.text_splitter import CharacterTextSplitter
import constants
# Decorate the initialization function with custom hash function
@st.cache_resource
def initialization():
  os.environ["OPENAI_API_KEY"] = constants.APIKEY
  # Enable to save to disk & reuse the model (for repeated queries on the same data)
  PERSIST = True
  Save_chat = True
  system_template = """ Please provide assistance based on the context below. 
  ----------------
  {context}
  ---------------"""
  messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template("Question:```{question}```")
  ]
  qa_prompt = ChatPromptTemplate.from_messages(messages)
  query = None
  if len(sys.argv) > 1:
      query = sys.argv[1]

  if PERSIST and os.path.exists("persist"):
      vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
      index = VectorStoreIndexWrapper(vectorstore=vectorstore)
  else:
      # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
      loader = DirectoryLoader("data/")
      if PERSIST:
        #   index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"},text_splitter=CharacterTextSplitter(chunk_size=2500, chunk_overlap=500)).from_loaders([loader])
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
      else:
          index = VectorstoreIndexCreator().from_loaders([loader])

  chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model="gpt-3.5-turbo-16k"),
      retriever=index.vectorstore.as_retriever(search_kwargs={"k": 10}),
      return_source_documents=True,combine_docs_chain_kwargs={"prompt": qa_prompt}
  )
  chat_history = []
  return chain, chat_history, Save_chat

def main():
  chain, chat_history, Save_chat=initialization()
  st.title("	:robot_face: Technical Support ChatBot")
  # Initialize chat history
  if "messages" not in st.session_state:  
      st.session_state.messages = []
  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

  if "user_id" not in st.session_state: 
      if user_id := st.text_input("Your Name:"):
        st.session_state.user_id = user_id
        st.success("Successfully logged in!")
        time.sleep(0.5)
        st.rerun()
  # React to user inputs
  else: 
    if prompt := st.chat_input("Ask something"):
      # Display user message in chat message container
      query= prompt
      st.chat_message("user").markdown(prompt)
      st.session_state.messages.append({"role": "user", "content": prompt})
      result = chain({"question": query, "chat_history": chat_history})
      response = f"{result['answer']}"
      # Display assistant response in chat message container
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
      # Add assistant response to chat history  
      for chunk in response.split(" "):
          full_response += chunk + " "
          time.sleep(0.05)
          # Add a blinking cursor to simulate typing
          message_placeholder.markdown(full_response + "▌")
      message_placeholder.markdown(full_response)
      st.session_state.messages.append({"role": "assistant", "content": response})
      chat_history.append((query, result['answer']))
      query = None
      if Save_chat: 
        with open(f"{st.session_state.user_id}.txt","a") as f:
          f.write(f"Question: {prompt}\nChatGPT: {result['answer']}\n\n")
    
    

if __name__ == "__main__":
    main()
