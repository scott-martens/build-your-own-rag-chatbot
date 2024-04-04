import streamlit as st
import os
import tempfile

from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from jina_rerank import JinaRerank

# Bilingual Models also available for:
# German/English: "jina-embeddings-v2-base-de"
# Chinese/English: "jina-embeddings-v2-base-zh"
# Spanish/English: "jina-embeddings-v2-base-es"
# And for programming languages plus English: "jina-embeddings-v2-base-code"
# See https://jina.ai/embeddings/ for more.

jina_embeddings_model_name = "jina-embeddings-v2-base-en"

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file, vector_store):
    if uploaded_file is not None:
        
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            # Larger chuck sizes work better, and Jina Embeddings supports
            # much larger ones than this and GPT 3.5 Turbo supports large
            # inputs too
            chunk_size = 15000,
            chunk_overlap = 3000
        )

        # Vectorize the PDF and load it into the Astra DB Vector Store
        pages = text_splitter.split_documents(docs)
        vector_store.add_documents(pages)  
        st.info(f"{len(pages)} pages loaded.")

# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. 
You prefer to use bulletpoints to summarize.
IMPORTANT: You must rely on the context information below, and as little as possible on
other knowledge.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("human", template)])
prompt = load_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-3.5-turbo',
        streaming=True,
        verbose=True
    )

chat_model = load_chat_model()

# Jina Reranker is currently only available in English
reranker = JinaRerank(
    model="jina-reranker-v1-base-en",
    jina_api_key=st.secrets["JINA_API_KEY"],
)

# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_vector_store():
    # Connect to the Vector Store
    vector_store = AstraDBVectorStore(
        embedding=JinaEmbeddings(model_name=jina_embeddings_model_name),
        collection_name="my_store",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 20}
    )
    return retriever
retriever = load_retriever()

def get_and_rank_docs(question):
    context_records = retriever.get_relevant_documents(question)
    reranked_items = reranker.rerank(query=question, documents=context_records, top_n=5)
    return [context_records[item['index']] for item in reranked_items]

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Draw a title and some markdown
st.title("Your personal Efficiency Booster")
st.markdown("""Generative AI is considered to bring the next Industrial Revolution.  
Why? Studies show a **37% efficiency boost** in day to day work activities!""")

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader('Upload a document for additional context', type=['pdf'])
        submitted = st.form_submit_button('Save to Astra DB')
        delete = st.form_submit_button('Delete contents of Astra DB')
        if submitted:
            vectorize_text(uploaded_file, vector_store)
        if delete:
            vector_store.clear()

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("What's up?"):
    
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling OpenAI's Chat Model

    inputs = RunnableMap({
        'context': lambda x: "\n\n".join([doc.page_content for doc in get_and_rank_docs(x['question'])]),
        'question': lambda x: x['question']
    })
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)