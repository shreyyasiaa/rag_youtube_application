import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
from pytube import YouTube
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import weaviate

# Streamlit app
st.title('YouTube Video Transcript QA with RAG')

# Input: YouTube URL
youtube_url = st.text_input("Enter YouTube video URL:")

if youtube_url:
    # Step 1: Extract Video Title
    video_id = youtube_url.split("v=")[-1]
    yt = YouTube(youtube_url)
    title = yt.title
    
    # Step 2: Extract Transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Step 3: Save Transcript as PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')

    pdf.set_font("Arial", size=12)
    for entry in transcript:
        text = entry['text'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, text)

    pdf_filename = f"{title}_transcript.pdf"
    pdf.output(pdf_filename)
    st.success(f"Transcript saved as {pdf_filename}")

    # Step 4: Load PDF and split into chunks
    loader = PyPDFLoader(pdf_filename, extract_images=True)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)

    # Step 5: Set up Weaviate and Embeddings
    WEAVIATE_CLUSTER="https://pnvqo1mrqaiqsm7tk3ag.c0.us-east1.gcp.weaviate.cloud"
    WEAVIATE_API_KEY="IShwZt5Pho2DBOeMCDFC7YYUtzAM01m2klTU"
    client = weaviate.Client(
        url=WEAVIATE_CLUSTER, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
    )

    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)

    retriever = vector_db.as_retriever()

    # Step 6: Set up RAG Chain
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    huggingfacehub_api_token="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"
    model = HuggingFaceHub(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature":1, "max_length":200}
    )

    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )

    # Input: Question from the user
    user_question = st.text_input("Ask a question about the video:")

    if user_question:
        # Get the answer from the RAG Chain
        answer = rag_chain.invoke(user_question)
        st.write("Answer:", answer)
