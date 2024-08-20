from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama

ollama_emb = OllamaEmbeddings(
    model="llama2",
)
new_vector_store = FAISS.load_local(
    "faiss_index", embeddings=ollama_emb, allow_dangerous_deserialization=True
)
def retriever(prompt):
    prompt1 = prompt.replace(" ", "")
    results = new_vector_store.similarity_search_with_score(
        prompt1, k=2
    )
    return results[0]


model = Ollama(model="llama2")
while True:
    prompt_now = input('please enter your question')
    prompt_real = retriever(prompt_now)[0].page_content + "\n"+"reponse according to the data on top and question below\n"+prompt_now
    response_text = model.invoke(prompt_real)
    print(response_text)
