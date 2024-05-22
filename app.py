import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

async def get_conversational_answer(retriever,input,chat_history):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    llm = ChatOllama(model="llama3")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ai_msg = rag_chain.invoke({"input": input, "chat_history": chat_history})
    return  ai_msg


def main():
    st.header('Chat with your PDF using ollama by llama3 and gemini')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history=[]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    embed_model = OllamaEmbeddings(model='mxbai-embed-large')

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',accept_multiple_files = True, type=['pdf'])
        if st.button('Process'):
            if docs is not None:
                os.makedirs('./data', exist_ok=True)
                for doc in docs:
                    save_path = os.path.join('./data', doc.name)
                    with open(save_path, 'wb') as f:
                        f.write(doc.getbuffer())
                    st.write(f'Processed file: {save_path}')
           
            with st.spinner('Processing'):
                loader = PyPDFDirectoryLoader("./data")
                documents = loader.load()
                vector_store = FAISS.from_documents(documents, embed_model)
                retriever=vector_store.as_retriever()
                if "retriever" not in st.session_state:
                    st.session_state.retriever = retriever
                st.session_state.activate_chat = True

            # Delete uploaded PDF files after loading
            for doc in os.listdir('./data'):
                os.remove(os.path.join('./data', doc))

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user",  "avatar" :'👨🏻', "content": prompt})
            retriever = st.session_state.retriever

            ai_msg = asyncio.run(get_conversational_answer(retriever,prompt,st.session_state.chat_history))
            st.session_state.chat_history.extend([HumanMessage(content=prompt), ai_msg["answer"]])
            cleaned_response=ai_msg["answer"]
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant",  "avatar" :'🤖', "content": cleaned_response})
        else:
            st.markdown('Upload your PDFs to chat')


if __name__ == '__main__':
    main()