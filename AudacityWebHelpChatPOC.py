import os
import streamlit as st
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)

#Some Global Variables for finding the database (for queries or refreshing it)
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

#url = st.text_input("Insert The website URL")  <-- uncomment (and comment next) if you want a DYNAMIC URL...
url = "https://manual.audacityteam.org/"

# Create OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

# Define a custom extractor function to extract the text content from the webpage
def extract_content(html):
    soup = Soup(html, "html.parser")
    content = soup.get_text()
    return content

def main():
    # Set the title and subtitle of the app
    st.title("🦜🔗 'How-To' Chat with Audacity Help")
    st.image('./assets/AudacityHelp.png')
    st.write('Refreshing your local Vector-Data is Optional but Recommended (takes ~30 seconds)...')
    if st.button("Refresh Data"):    
        with st.spinner('In Progress...'):
            # Load data from the specified URL
            loader = RecursiveUrlLoader(
                url=url, max_depth=3, extractor=extract_content
            )

            data = loader.load()

            # Split the loaded data
            text_splitter = CharacterTextSplitter(separator='\n', 
                                            chunk_size=1000, 
                                            chunk_overlap=40)

            docs = text_splitter.split_documents(data)

            # Filter complex metadata from the documents
            filtered_documents = filter_complex_metadata(docs)

            # Create a Chroma vector database from the documents
            vectordb = Chroma.from_documents(documents=filtered_documents, 
                                            embedding=openai_embeddings,
                                            persist_directory=DB_DIR)

            vectordb.persist()
        st.success("Data Refreshed!")
    st.subheader('Ask a Question, get the Answer with Source Document Links.')
    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):

        vectordb = Chroma(embedding_function=openai_embeddings, persist_directory=DB_DIR)

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        # Run the prompt and return the response
        response = qa(prompt)
        st.subheader("Answer:")
        st.write(response["result"])
        st.subheader("Top 3 Sources Used:")
        for source_doc in response["source_documents"]:
            title = source_doc.metadata["title"]
            sourceLink = source_doc.metadata["source"]
            stringified = "["+title+"]"+"("+sourceLink+")"
            st.write(stringified)

if __name__ == '__main__':
    main()