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
    scores = {}
    prompt1 = prompt.split(" ")
    for word in prompt1:
        results = new_vector_store.similarity_search_with_score(
            word, k=1,
        )
        scores[results[0][0].page_content] = results[0][1]
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=False)
    print(scores)
    return scores[0][0]


model = Ollama(model="llama2")
while True:
    prompt_now = input('please enter your question')
    prompt_real = "the person asking this question visited this website" + retriever(prompt_now) + "\n" + "reponse according to the data on top and question below\n" + prompt_now
    response_text = model.invoke(prompt_real)
    print(response_text)
