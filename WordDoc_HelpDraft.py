import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

#Some Global Variables for finding the database (for queries or refreshing it)
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "WordDocVectorDB")
DOC_PATH: str = os.path.join(ABS_PATH, "HelpDoc.docx")
#set the 

# Create OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

def main():
    # Set the title and subtitle of the app
    st.title("🦜🔗 Chat With A Word Doc")
    st.image('./assets/Wiki-Pikture.png')
    st.write('Create your local Vector-Data at least once (takes ~30 seconds)...')
    if st.button("Refresh Data"):
        with st.spinner('In Progress...'):
            #Load data from the WORD DOC
            loader = UnstructuredWordDocumentLoader(
                DOC_PATH, 
                mode="elements",
                break_on_headings=True, 
                break_on_paragraphs=True
            )
            docs = loader.load()

            # Filter complex metadata from the documents
            filtered_documents = filter_complex_metadata(docs)

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=filtered_documents, 
                                        embedding=openai_embeddings,
                                        persist_directory=DB_DIR)

        vectordb.persist()
        st.success("Data Refreshed!")
    st.subheader('Ask a Question, get the Answer with Source Document Links.')
    user_question = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):

        # Reference the Refreshed DB.
        vectordb = Chroma(embedding_function=openai_embeddings, persist_directory=DB_DIR)

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 270})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=retriever, 
            return_source_documents=True)

        # Run the prompt and return the response
        response = qa(user_question)
        st.subheader("Answer:")
        st.write(response["result"])
        st.subheader("Top Sources Used:")
        for source_doc in response["source_documents"]:
            category = source_doc.metadata["category"]
            text = source_doc.page_content
            st.write("Category: "+category+
                     "Content: "+text+
                     " ")
 
if __name__ == '__main__':
    main()