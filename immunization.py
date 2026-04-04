import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate

import logging
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

TEST_GROQ       = True   # Change to False will use OpenAI LLM

load_dotenv()
if TEST_GROQ:
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
else:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



# 1. 開啟日誌的「螢幕輸出」功能，並預設為 WARNING (避免被其他套件的碎碎念洗版)
logging.basicConfig(
    level=logging.WARNING, 
    format='%(levelname)s:%(name)s: %(message)s'
)

# 2. 單獨指定把 MultiQueryRetriever 的音量調大到 INFO
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Load PDF
print("System initializing: Loading Immunisation Handbook PDF...")
loader = PyPDFLoader("immunisation_handbook.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) # this chunk_size controls max # of words for one paragraph
splits = text_splitter.split_documents(docs)

if TEST_GROQ:
    print("===================================> Current mode: Groq + Google")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    my_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(embedding_function=my_embeddings, persist_directory="./chroma_db_gemini")

    if not vectorstore.get()['ids']:
        total_splits = len(splits)
        print(f"⏳ 發現新文件！準備將 {total_splits} 個區塊上傳給 Google 算向量...")
        
        batch_size = 15  # 安全極限：每次只拿 15 個區塊
        i = 0
        
        # 3. 加入無敵防護罩的迴圈
        while i < total_splits:
            batch = splits[i : i + batch_size]
            try:
                # 嘗試寫入資料庫
                vectorstore.add_documents(batch)
                i += batch_size
                print(f"✅ 上傳進度: {min(i, total_splits)} / {total_splits}")
                
                # 正常休息 10 秒 (15個/10秒 = 90個/分鐘，完美避開 100個的限制)
                time.sleep(10) 
                
            except Exception as e:
                # 如果發生錯誤，檢查是不是 429 頻率限制
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("⚠️ 撞到 Google 每分鐘 100 次的限制了！強迫喝水休息 60 秒後自動重試...")
                    time.sleep(60) # 睡一分鐘等額度重置，醒來後會「自動重試」剛剛失敗的那批！
                else:
                    # 如果是其他未知的錯誤，才讓程式停止
                    raise e
                    
        print("🎉 資料庫完全存檔完畢！以後執行就不會再呼叫 API 了！")
    else:
        print("⚡ 發現本機已存檔的資料庫，免連網，一秒讀取完畢！")
else:
    print("===================================> Current mode: OpenAI")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    my_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100)
    vectorstore = Chroma(persist_directory="./chroma_db_openai", embedding_function=my_embeddings)

    if not vectorstore.get()['ids']:
        print("💡 偵測到新文件，正在計算 OpenAI 向量並存入硬碟...")
        # 只有沒資料時，才執行這行「耗時且要付費」的動作
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=my_embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print("✅ 向量存檔完畢！")
    else:
        print("⚡ 發現本機存檔，已直接載入資料庫，不再呼叫 OpenAI Embedding API。")



# ---------------------------------------------------------
# 第一把武器：傳統關鍵字檢索器 (BM25)
# 直接讀取切好的文本，鎖定精準的關鍵字比對
# ---------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4  # 讓它抓 4 段關鍵字最吻合的文本

# ---------------------------------------------------------
# 第二把武器：向量語意檢索器 (Vector)
# 保留我們剛剛設定好的 MMR 演算法，負責抓取跨章節的語意關聯
# ---------------------------------------------------------
vector_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 4, "fetch_k": 20}
)

# ---------------------------------------------------------
# 終極融合：混合檢索器 (Ensemble)
# ---------------------------------------------------------
# weights=[0.5, 0.5] 代表兩者的搜尋結果權重各佔一半
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an expert medical researcher. Your task is to generate 3 distinct search queries to find 
information in a medical handbook.
Crucially, medical handbooks separate "general rules" (like fever) from "specific vaccine rules" (like MMR age limits).
Please generate exactly 3 queries based on the original question:
1. A query asking ONLY about the specific vaccine and age group (ignore the symptoms).
2. A query asking ONLY about the specific symptoms/conditions (including synonyms like "low-grade fever", 
"minor illness") as a general contraindication for ANY vaccination.
3. A query combining them or rephrasing the original question.

Original Question: {question}

Provide the queries separated by newlines, with no numbering or extra text."""
)

ultimate_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,  # <--- 讓它指揮雙武器去搜尋
    llm=llm,
    prompt=QUERY_PROMPT
)

# 接下來，原本的 create_retrieval_chain(retriever=retriever, ...) 都不用改！

#base_retriever = vectorstore.as_retriever(
#    search_type="mmr", 
#    search_kwargs={"k": 6, "fetch_k": 20}
#)
#
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
## 告訴 AI 醫療手冊的編排邏輯，強迫它把問題拆開
#QUERY_PROMPT = PromptTemplate(
#    input_variables=["question"],
#    template="""You are an expert medical researcher. Your task is to generate 3 distinct search queries to find information in a medical handbook.
#Crucially, medical handbooks separate "general rules" (like fever) from "specific vaccine rules" (like MMR age limits).
#Please generate exactly 3 queries based on the original question:
#1. A query asking ONLY about the specific vaccine and age group (ignore the symptoms).
#2. A query asking ONLY about the specific symptoms/conditions as a general contraindication for ANY vaccination.
#3. A query combining them or rephrasing the original question.
#
#Original Question: {question}
#
#Provide the queries separated by newlines, with no numbering or extra text."""
#)
#
#
## 2. 將基礎檢索器與你的 LLM 綁定，升級成 MultiQueryRetriever
## (注意：這裡的 'llm' 變數請替換成你在程式碼中定義 gpt-4o-mini 的那個變數名稱)
#retriever = MultiQueryRetriever.from_llm(
#    retriever=base_retriever,
#    llm=llm,
#    prompt=QUERY_PROMPT
#)

from langchain_core.prompts import ChatPromptTemplate

# 這是最後用來產出答案的 QA Prompt
#qa_prompt = (
#"""You are a helpful Clinical Advisor. Use the following pieces of retrieved context to answer the question. 
#    
#    CRITICAL INSTRUCTION FOR MEDICAL HANDBOOKS:
#    The context may contain "general vaccination rules" in one chunk, and "specific vaccine rules" in another. 
#    You are explicitly authorized to synthesize them. For example, if Chunk A says "Condition X is not a general 
#    contraindication", and Chunk B says "Vaccine Y can be given at age Z", you must combine these facts to answer 
#    questions about giving Vaccine Y to a person of age Z with Condition X.
#    
#    If you still cannot find the answer after synthesizing, say "Not mentioned in the guidelines."
#    
#    Context: {context}"""
#)

qa_prompt = (
"""You are a helpful Clinical Advisor. Use the following pieces of retrieved context to answer the question. 
    
    CRITICAL INSTRUCTION FOR MEDICAL HANDBOOKS:
    The context may contain "general vaccination rules" in one chunk, and "specific vaccine rules" in another. 
    You are explicitly authorized to synthesize them. 
    
    Context: {context}
    
    Let's think step-by-step:
    1. First, state what the context says about the specific vaccine (MMR) and the patient's age.
    2. Second, state what the context says about general vaccination rules regarding the patient's condition (e.g., mild fever, minor illness, low-grade fever).
    3. Finally, synthesize these facts to provide a clear recommendation.
    
    If the context completely lacks either the age rule or the fever rule, conclude with "Not mentioned in the guidelines."
"""
)

# 確保你的 chain 是用這個更新過的 qa_prompt
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# retrieval_chain = create_retrieval_chain(ultimate_retriever, question_answer_chain)

prompt = ChatPromptTemplate.from_messages([
    ("system", qa_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
#rag_chain = create_retrieval_chain(ultimate_retriever, question_answer_chain)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n[ OK ] System Ready!")
user_question = "Can a 6-month-old infant with a mild fever receive the MMR vaccine?"
#"Can a pregnant woman receive the Varicella (chickenpox) vaccine?" 

print(f"Clinical Advisor asking: {user_question}\n")

response = rag_chain.invoke({"input": user_question})

print("=== AI Advisor Answer ===")
print(response["answer"])

print("\n=== Sources Triggered (Summary) ===")
# 為了避免變數名稱錯亂，我們統一把它們裝進一個變數
retrieved_docs = response.get("context", [])

# 1. 先印出簡短的摘要清單 (前 60 個字)
for i, doc in enumerate(retrieved_docs):
    page_num = doc.metadata.get('page', 'Unknown')
    # 將換行符號替換掉，讓摘要列表看起來更整齊
    short_content = doc.page_content[:60].replace('\n', ' ')
    print(f"Source {i+1} (Page {page_num}): {short_content}...")

print("\n" + "="*50)
print("=== Detailed Full Text for Debugging ===")
print("="*50)

# 2. 再印出完整的純文字內容供你檢查
for i, doc in enumerate(retrieved_docs):
    print(f"\n=== Source {i+1} (Python認定的頁碼: {doc.metadata.get('page', 'Unknown')}) ===")
    print(doc.page_content) 
    print("-" * 40)
