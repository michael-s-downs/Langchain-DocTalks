import os
import shutil
from typing import Any, List, Dict, Optional
from langchain_core.embeddings import Embeddings
import streamlit as st
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import json
import requests
import numpy as np
from langchain_core.callbacks.manager import CallbackManagerForChainRun

#VARIABLES-SETUP SECTION:  Config (non-secure info) and Environment (Secured info) 
# load config: update or create Atlassian Wiki username and its api_key in config.json TODO:  move api_key to Environment
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

#Other Global Variables for finding and creating the docker-onboard ChromaDB (for queries or refreshing it)
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "AspirentWikiDB")
url = "https://aspirentconsulting.atlassian.net/wiki"

#CUSTOM CLASSES, WRAPPERS, EXTENSIONS, DEFs SECTION. MOST WRAPPERS ARE TO CONSUME OUR OWN API_CLIENT:
class CustomRetrievalQA(BaseRetrievalQA):
    def __init__(
            self, 
            api_client: Optional[Any] = None, 
            llm: Optional[Any] = None, 
            chain_type: Optional[str] = None, 
            retriever: Optional[Any] = None, 
            return_source_documents: Optional[bool] = None, 
            **kwargs):
        super().__init__(**kwargs)

        self.api_client = api_client
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.return_source_documents = return_source_documents

    #Pydantic Voodoo to disable its Pedantry for this class which inherits from a Pydantic Model
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        
    #Existing abstract methods and dummy attributes that we must 'implement' in our wrapper but don't actually need...
    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get documents to do question answering over."""

    def _aget_docs(self, query: str) -> List[Document]: 
         pass

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,  #we do not use the run_manager but it is part of the signature...
    ) -> Dict[str, Any]:
        
        question = inputs[self.input_key]
        docs = self.retriever.get_relevant_documents(question)

        # Generate context from documents
        context = "\n\n".join(doc.page_content for doc in docs)

        #this should give us back a dictionary with an answer and a blank list of Documents
        answer = self.api_client.chat_completion(question, context)
        
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}    

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

    def generate_vectors(self, texts):
        headers = {
            "Ocp-Apim-Subscription-Key": f"{self.api_key}"
        }
        data = {"input": texts}
        
        response = requests.post(f"{self.api_url}/deployments/{self.api_embed_depId}/embeddings?api-version={self.api_version}", json=data, headers=headers)
        response.raise_for_status()
        
        vectors = response.json()["data"]
        return [v["embedding"] for v in vectors]
		
    def chat_completion(self, user_question, context):
        headers = {
            "Ocp-Apim-Subscription-Key": f"{self.api_key}"
        }
        request_body = {
        "messages": [
            {
                "role": "system",
                "content": f"Use the provided context to answer the user question. Do not make up answers if you don't know, just say 'I don't know'.  CONTEXT: {context}"  
            },
            {"role": "user", "content": user_question}
        ],
        "max_tokens": 800,
        "temperature": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.95,
        "stop": None
    }
        response = requests.post(f"{self.api_url}/deployments/{self.api_mod_depId}/chat/completions?api-version={self.api_version}", json=request_body, headers=headers)
        response.raise_for_status()        

        # Get the "choices" list from response
        choices = response.json()["choices"]
    
        # We just need the first choice
        choice = choices[0]
    
        # Extract the message content 
        answer = choice["message"]["content"]

        return {
            "result": answer,
            "source_documents": []
        }


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

#########################
#MAIN RUNTIME FUNCTION
#########################
def main():

    #Create Custom api_client
    api_client = TCCC_AzureOpenAI_APIClient(
    api_key=AZURE_OPENAI_API_SUBSCRIPTION_KEY, 
    api_url=AZURE_OPENAI_API_URL, 
    api_mod_depId = AZURE_OPENAI_API_DEPLOYMENT_ID,
    api_embed_depId = AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_ID,
    api_version = AZURE_OPENAI_API_VERSION
    )

    #Create Custom Embeddings Wrapper that consumes api_client
    custom_embeddings = CustomEmbeddings(api_client=api_client)

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

            vectordb = Chroma.from_documents(documents=docs, 
                                embedding=custom_embeddings,
                                persist_directory=DB_DIR)

            vectordb.persist()
        st.success("Data Refreshed!")
    st.subheader('Ask a Question, get the Answer with Source Document Links.')
    user_question = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):

        vectordb = Chroma(embedding_function=custom_embeddings, persist_directory=DB_DIR)

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 30})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-4-turbo-preview')

        # Create a CUSTOM RetrievalQA from the model and retriever
        qa = CustomRetrievalQA.from_chain_type(api_client=api_client, llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        # Run the prompt and return the response (this is where it makes the call out, now to API instead of Service)
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