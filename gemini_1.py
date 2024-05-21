import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from dotenv import load_dotenv
load_dotenv()


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    #llm = ChatOllama(model="llama3")
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


async def user_input(db,user_question):
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response= chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    return response


def main():

    st.header('Chat with your PDF using ollama by llama3')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    #embed_model = OllamaEmbeddings(model='mxbai-embed-large')
    embed_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")


    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',accept_multiple_files = True, type=['pdf'])
        if st.button('Process'):
            if docs is not None:
                for doc in docs:
                    with NamedTemporaryFile(dir='.', suffix='.pdf',delete=False) as f:
                        f.write(doc.getbuffer())
                        st.write(f'Processed file: {f.name}')
           
            with st.spinner('Processing'):
                loader = PyPDFDirectoryLoader(".")
                documents = loader.load()
                vector_store = FAISS.from_documents(documents, embed_model)
                if "vector_store" not in st.session_state:
                    st.session_state.vector_store = vector_store
                st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "avatar" :'👨🏻',"content": prompt})

            vector_store = st.session_state.vector_store
            pdf_response=asyncio.run(user_input(vector_store,prompt))
            cleaned_response=pdf_response["output_text"]
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", "avatar" :'🤖',"content": cleaned_response})
        else:
            st.markdown('Upload your PDFs to chat')


if __name__ == '__main__':
    main()