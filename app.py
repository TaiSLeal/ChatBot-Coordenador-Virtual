import streamlit as st
from dotenv import load_dotenv
import qdrant_client
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
import os
from langchain.embeddings.openai import OpenAIEmbeddings

def get_vectorstore(cursoSelecionado):

    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embeddings = OpenAIEmbeddings()

    if cursoSelecionado == "Engenharia química":
        collection_name = os.getenv("QDRANT_COLLECTION_NAME1")
    else: 
        collection_name = os.getenv("QDRANT_COLLECTION_NAME2")
    vectorstore = Qdrant(
        client = client,
        collection_name = collection_name,
        embeddings=embeddings
    )
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="CHATBOT",page_icon="https://www.eng.ufba.br/sites/eng.ufba.br/themes/ufba_bootstrap/favicon.ico")
    st.header("Coordenador Virtual")

    cursoSelecionado = st.sidebar.selectbox(label='selecione o curso',options=["Engenharia química", "Engenharia de controle e automação"])

    vectorstore = get_vectorstore(cursoSelecionado)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever = vectorstore.as_retriever()
    )
    user_question = st.text_input("escreva aqui sua pergunta")
    if user_question:
        st.write(f"Pergunta: {user_question}")
        answer = qa.run(f"oi, preciso que você me ajude a reponder as duvidas dos alunos de engenharia da politecnica UFBA, você será um coordenador virtual, e vai responder os alunos apenas com os documentos fornecidos, dé preferência aos documentos mais atualizados como por exemplo o guia do estudante supac. Caso não encontre alguma resposta no material seja gentil e peça desculpa e oriente o aluno a procurar a coordenação do curso, informe documentações necessárias e as ações necessárias para realizar o procedimento. Seja verdadeiro e objetivo e não invente informações. A pergunta é {user_question}")
        st.write(f"Resposta: {answer}")


if __name__ == '__main__':
    main()
