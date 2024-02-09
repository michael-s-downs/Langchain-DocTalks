import os
import streamlit as st
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import json

# load config: update or create username and api_key in config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
userName = config["username"]
api_key = config["api_key"]


# Load environment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

#Some Global Variables for finding the database (for queries or refreshing it)
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "AspirentWikiDB")
url = "https://aspirentconsulting.atlassian.net/wiki"


# Create OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

#In order to apply a custom extractor to our ConfluenceLoader, we need to create/extend our own custom ConfluenceLoader...
class CustomConfluenceLoader(ConfluenceLoader):
    def process_page(self, page_id, space_key, content_format, include_attachments, *args, **kwargs):
        # Call the parent process_page method to retrieve the page content
        page_content = super().process_page(page_id, space_key, content_format, include_attachments, *args, **kwargs)

        oneDocList = [page_content]

        #filter any bad Document Metadata
        filteredPage_ContentList = filter_complex_metadata(oneDocList)        

        #take back the document from the list
        filteredDocument = filteredPage_ContentList[0]

        #get the metadata from the filtered Document
        filteredDocument_metadata = filteredDocument.metadata

        #make sure we only get the html from the Document object returned...
        html_content = filteredDocument.page_content

        # Apply custom extraction logic to the page content
        extracted_content = extract_content(html_content)

        # Finally, we need to turn it back into a Document object for further down-stream processing
        new_document = Document(page_content=extracted_content, metadata=filteredDocument_metadata)
        return new_document

#Define a custom extractor function to extract the text content from the webpage -- CURRENTLY USED
def extract_content(html):
    soup = Soup(html, "html.parser")
    content = soup.get_text()
    return content

#EXPERIMENTAL custom extractor function to extract text AND images from the webpage -- FUTURE USE
def extract_content_and_images(html):
    soup = Soup(html, "html.parser")
    text_content = soup.get_text()
    image_tags = soup.find_all('img')
    return text_content, image_tags

def main():
    # Set the title and subtitle of the app
    st.title("🦜🔗 Chat with Query Tool Confluence")
    st.image('./assets/Wiki-Pikture.png')
    st.write('Re/Create your local Vector-Data...')
    if st.button("Refresh Data"):    
        with st.spinner('In Progress...'):

            loader = CustomConfluenceLoader(
                url=url,
                username=userName,
                api_key=api_key
            )
            data = loader.load(space_key="CS", include_attachments=False, limit=50)  #starting with include attachments as false

            # Split the loaded data
            text_splitter = CharacterTextSplitter(separator='\n', 
                                            chunk_size=1000, 
                                            chunk_overlap=40)

            docs = text_splitter.split_documents(data)

            # Filter complex metadata from the documents
            #filtered_documents = filter_complex_metadata(docs)

            # Create a Chroma vector database from the documents
            vectordb = Chroma.from_documents(documents=docs, 
                                            embedding=openai_embeddings,
                                            persist_directory=DB_DIR)

            vectordb.persist()
        st.success("Data Refreshed!")
    st.subheader('Ask a Question, get the Answer with Source Document Links.')
    user_question = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):

        vectordb = Chroma(embedding_function=openai_embeddings, persist_directory=DB_DIR)

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 30})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        #tokenLength = qa.extra.extracted_content() 
        #st.subheader("Pre-Submission Token Check:")
        #st.write("token length: "+ tokenLength)
        #st.write("token content: " + qa.extracted_content())

        # Run the prompt and return the response
        response = qa(user_question)
        st.subheader("Answer:")
        st.write(response["result"])
        st.subheader("Top Sources Used:")
        for source_doc in response["source_documents"]:
            title = source_doc.metadata["title"]
            sourceLink = source_doc.metadata["source"]
            stringified = "["+title+"]"+"("+sourceLink+")"
            st.write(stringified)

if __name__ == '__main__':
    main()