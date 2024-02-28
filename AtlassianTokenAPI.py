import os
import shutil
from typing import List
from langchain_core.embeddings import Embeddings
import streamlit as st
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import json
import requests
import numpy as np

# load config: update or create username and api_key in config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
userName = config["username"]
api_key = config["api_key"]

# Load environment variables from .env file (Optional)
load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_API_SUBSCRIPTION_KEY")
AZURE_OPENAI_API_URL = os.getenv("AZURE_OPENAI_API_URL")
AZURE_OPENAI_API_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_ID")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_ID=os.getenv("AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_ID")

#Some Global Variables for finding the database (for queries or refreshing it)
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "AspirentWikiDB")
url = "https://aspirentconsulting.atlassian.net/wiki"


# Create OpenAI embeddings  WE WILL REPLACE THIS WITH A CUSTOM WRAPPER OBJECT THAT TAKES OUR API_CLIENT AND USES IT...
openai_embeddings = OpenAIEmbeddings()

class CustomEmbeddings(Embeddings):

    def __init__(self, api_client):
        self.api_client = api_client

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        vectors = self.api_client.generate_vectors(texts)
        return vectors

    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        #texts = [doc.page_content for doc in documents]  
        return self.embed(documents)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

class TCCC_AzureOpenAI_APIClient:
    def __init__(self, api_key, api_url, api_mod_depId, api_embed_depId, api_version):
        self.api_key = api_key 
        self.api_url = api_url
        self.api_mod_depId = api_mod_depId
        self.api_embed_depId = api_embed_depId
        self.api_version = api_version

    #sample URL:  https://apim-emt-aip-prod-01.azure-api.net/openai/deployments/{embed-deployment-id}/embeddings?api-version={api-version}
    def generate_vectors(self, texts):
        headers = {
            "Ocp-Apim-Subscription-Key": f"{self.api_key}"
        }
        data = {"input": texts}
        
        response = requests.post(f"{self.api_url}/deployments/{self.api_embed_depId}/embeddings?api-version={self.api_version}", json=data, headers=headers)
        response.raise_for_status()
        
        vectors = response.json()["data"]
        return [v["embedding"] for v in vectors]
		
	#sample URL:  https://apim-emt-aip-prod-01.azure-api.net/openai/deployments/{mod-deployment-id}/chat/completions?api-version={api-version}
    #def chat_completion():	#<-- complete this definition later

#Create Custom api_client
api_client = TCCC_AzureOpenAI_APIClient(
    api_key=AZURE_OPENAI_API_SUBSCRIPTION_KEY, 
    api_url=AZURE_OPENAI_API_URL, 
    api_mod_depId = AZURE_OPENAI_API_DEPLOYMENT_ID,
    api_embed_depId = AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_ID,
    api_version = AZURE_OPENAI_API_VERSION
    )



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
    st.title("🦜🔗 CHAT: OQT Wiki via COKE-API")
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

            if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
                shutil.rmtree(DB_DIR)

            # # Create a Chroma vector database from the documents
            # vectordb = Chroma.from_documents(documents=docs, 
            #                                 embedding=openai_embeddings,
            #                                 persist_directory=DB_DIR)
            
            # NEW SECTION 
            custom_embeddings = CustomEmbeddings(api_client=api_client)
            vectordb = Chroma.from_documents(documents=docs, 
                                embedding=custom_embeddings,
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
        llm = ChatOpenAI(model_name='gpt-4-turbo-preview')

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

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