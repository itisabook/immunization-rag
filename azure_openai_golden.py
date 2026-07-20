import json
from openai import OpenAI  # 注意：這裡是 OpenAI 而不是 AzureOpenAI
import pandas as pd
import os
import datetime
import time
from tenacity import retry, stop_after_attempt, wait_fixed
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")

start_time = time.time()  # 紀錄開始時間
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 從你剛才的 Sample Code 複製過來的資訊
endpoint = "https://ai-team-02-hack.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5-mini" # 這是你的部署名稱

BASE_TRANSCRIPT_PATH = r"c:\Users\Owner\Documents\compsci714\hackathon\immunization_guidelines\output20260400\local-folder"
input_file = "test_benchmark_100_20260530_003852.csv"
#"test_benchmark_25_20260522_025026_new.csv"

#"test_benchmark_25_20260522_041352_new.csv"
md_file = "NZ_Immunisation_Handbook_2026_by_marker.md"
#"immunisation-handbook-2026-v2.md"
output_file = f"golden_{Path(input_file).stem}_{timestamp}.csv"

# 2. 初始化 Client
# 注意：使用 Project Endpoint 時，API Key 放在這裡通常就能跑
client = OpenAI(
    base_url=endpoint,
    api_key=my_api_key
)

def librarian_agent(transcript, chapters_list):
    # 將清單轉為字串供 Prompt 使用
    options = "\n".join(chapters_list)
    
    prompt = """
    你是一位專業的醫療圖書館員。你的任務是根據「診間對話」內容，從下方清單中挑選出最相關的章節標題，以便後續進行臨床正確性稽核。

    ### 挑選邏輯：
    1. 核心優先：參考章節# Commonly used abbreviations和章節# Glossary of vaccine brand names and abbreviations來辨識對話內容提到的疫苗/疾病，然後挑選對應章節（如Varicella 對應到# 24 Varicella (chickenpox)）。
    2. 補種必選：涉及「漏打」、「補打」、「遲到」、「沒帶記錄卡」或「海外搬回」，必選 "# Appendix 2: Planning immunisation catch-ups"。
    3. 情境化觸發：
        - # 2 Processes for safe immunisation:
            Select this chapter ONLY IF the query covers the topics as listed below:
            - ## 2.1 Pre-vaccination (Clinical assessment, Screening, Contraindications, Precautions, Co-administration of vaccines, Cold Chain concerns, Fever/Acute illness at the time of vaccination, Informed consent)
            - ## 2.2 Vaccine administration (Injection techniques: Intramuscular/Subcutaneous/Intradermal, Needle size/length, Anatomical sites: Deltoid / Vastus lateralis / Ventrogluteal, Positioning, Multiple injections)
            - ## 2.3 Post-vaccination (Observation period, Management of Anaphylaxis, Fainting/Syncope, Common side effects, Post-vaccination advice/education)
        - # 3 Vaccination questions and addressing concerns：
            涉及疫苗時程安排、嬰幼兒接種問題、過敏與疾病者是否能打針的判斷、活毒疫苗的傳播風險、和疫苗接種的常見誤解與擔憂。
        - # 4 Immunisation of special groups: 
            Select this chapter ONLY IF the query covers multiple special groups as listed below:
            - ## 4.1 Pregnancy and lactation (Maternal immunisation, breastfeeding)
            - ## 4.2 Infants with special considerations (Preterm, low birthweight, infants of mothers on biologics)
            - ## 4.3 Immunocompromised individuals (Immunosenescence, Immunosuppression, Immunomodulation, Cancer, Chemotherapy, HIV, Steroids, Biologics, Organ transplant, HSCT, Asplenia; also includes vaccination for Household/Close contacts to protect these patients)
            - ## 4.4 Chronic kidney disease (Dialysis, renal failure)
            - ## 4.5 Chronic liver disease
            - ## 4.6 Other special groups (Cochlear implants, CSF leak, Diabetes, Chronic pulmonary/heart disease, Close quarters/Institutions, Correctional facilities, MSM)
            - ## 4.7 Immigrants and refugees
            - ## 4.8 Occupation-related vaccination (Healthcare workers, ECE staff (Early Childhood Education), Emergency services, Laboratory/Sewage workers, Animal handlers, Border officials, Occupational immunization policy)
            - ## 4.9 Travel
        - # Appendix 3: Immunisation standards for vaccinators and guidelines for organisations offering immunisation services：
            涉及「醫療錯誤（Vaccination Error）」、「過期疫苗施打」、「診所通報流程」或「護士專業責任」時【必選】。此章節用於評估診所的處置是否符合專業標準。
        - # Appendix 5: Immunisation certificate： 涉及免疫證明、免疫登記系統、健康手冊。
    4. 輸出規範：
        - 請精選 1-3 個章節，除非絕對必要，否則不得超過 4 個。
        - Even if you are not 100% sure, provide the best possible matches. If the query is clinical, DO NOT return an empty list.
        - Note：回傳的標題必須與「章節清單」中的字元、符號（包括星號 *）完全一致，否則索引會失敗。

    【章節清單】:
    [OPTIONS_LIST]

    請嚴格回傳 JSON 格式：
    {
        "selected_chapters": ["標題A", "標題B"]
    }
    """

    prompt = prompt.replace("[OPTIONS_LIST]", options)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": transcript}],
        response_format={ "type": "json_object" },
        reasoning_effort="medium",
    )
    return json.loads(response.choices[0].message.content)["selected_chapters"]

def extract_chapter_content(selected_chapter_key, index_map, full_text):
    """
    根據 AI 選出的 key (例如 'Chapter 7')，從 full_text 擷取內容
    """
    # 1. 整理索引：確保我們知道章節的順序
    # 假設 index_map 格式為 {"Chapter 7": 1234, "Chapter 13": 5678, ...}
    sorted_keys = sorted(index_map.keys(), key=lambda x: index_map[x])
    
    if selected_chapter_key not in index_map:
        return f"找不到 {selected_chapter_key} 的內容。"

    start_pos = index_map[selected_chapter_key]
    
    # 2. 尋找結束位置 (即下一個章節的起點)
    try:
        current_idx = sorted_keys.index(selected_chapter_key)
        next_chapter_key = sorted_keys[current_idx + 1]
        end_pos = index_map[next_chapter_key]
    except (IndexError, ValueError):
        # 如果是最後一章，就抓到文件結尾
        end_pos = len(full_text)
    
    # 3. 擷取並回傳
    return full_text[start_pos:end_pos].strip()

def clean_references(key_name, text):
    # 定義常見的參考文獻開頭字樣
    ref_headers = [
        "\n# References", 
        "\n## References", 
        "\n### References",
        "\n#### References",
        "\n> References",
    ]
    
    clean_text = text
    text_length = len(text)
    
    # 如果內容本來就很短（例如小於 500 字），可能根本沒必要切，避免誤殺
    #if text_length < 500:
    #    return text

    for header in ref_headers:
        # 使用 rfind 從後面開始找，通常 References 都在最後面
        last_idx = clean_text.rfind(header)
        if last_idx != -1:
            # 安全檢查：參考文獻通常出現在章節的最後 30% 內容中
            # 如果發現關鍵字出現在很前面（例如前 20%），那很有可能是誤判，不予裁切
            if last_idx > (text_length * 0.3):
                print(f"已裁切 {key_name} 的參考文獻部分 (位置: {last_idx})")
                clean_text = clean_text[:last_idx]
                break 
                
    return clean_text

def get_combined_context(selected_keys, index_map, full_text, chapters_list):
    combined_content = ""
    
    # 1. 先找出所有「真正存在」於 index_map 中的全稱標題
    matched_full_keys = []
    
    for ai_key in selected_keys:
        ai_key_clean = ai_key.lower().strip()
        
        # 在手冊章節清單中尋找最接近的標題
        found_match = None
        for actual_chapter in chapters_list:
            # 只要 AI 給的字串被包含在標題內（例如 "HPV" 在 "Chapter 10: HPV" 裡）
            # 或者標題被包含在 AI 的字串內，就算匹配成功
            if ai_key_clean in actual_chapter.lower() or actual_chapter.lower() in ai_key_clean:
                found_match = actual_chapter
                break
        
        if found_match and found_match in index_map:
            matched_full_keys.append(found_match)
        else:
            print(f"警告:找不到匹配章節: {ai_key}")

    # 確保按照手冊順序排序
    sorted_selected = sorted(matched_full_keys, key=lambda x: index_map.get(x, 0))
    for key in sorted_selected:
        start = index_map[key]
                
        # 找到下一個章節的起點
        try:
            current_idx = chapters_list.index(key)
            next_key = chapters_list[current_idx + 1]
            end = index_map.get(next_key, len(full_text))
        except:
            # 如果是最後一章或出錯，就取到文件末尾
            end = len(full_text)

        raw_segment = full_text[start:end]
        
        if len(raw_segment.strip()) < 50:
            print(f"警告: 章節 {key} 抓取內容過短")
            
        # 移除參考文獻
        useful_segment = clean_references(key, raw_segment)

        # 組合內容
        combined_content += f"\n# SECTION CONTENT: {key}\n" 
        combined_content += useful_segment
        combined_content += f"\n\n"

    print(f"Total Context Length: {len(combined_content)}")
    return combined_content

# --- 1. 前置準備：讀取大檔案並建立索引 ---
with open(md_file, 'r', encoding='utf-8') as f:
    full_text = f.read()

# 建立索引地圖
chapters = [
    "# 1 General immunisation principles",
    "# 2 Processes for safe immunisation",
    "# 3 Vaccination questions and addressing concerns",
    "# 4 Immunisation of special groups",
    "# 5 Coronavirus disease (COVID-19)",
    "# 6 Diphtheria",
    "# 7 Haemophilus influenzae type b (Hib) disease",
    "# 8 Hepatitis A",
    "# 9 Hepatitis B",
    "# 10 Human papillomavirus",
    "# 11 Influenza",
    "# 12 Measles",
    "# 13 Meningococcal disease",
    "# 14 Mpox (orthopoxvirus)",
    "# 15 Mumps",
    "# 16 Pertussis (whooping cough)",
    "# 17 Pneumococcal disease",
    "# 18 Poliomyelitis",
    "# 19 Respiratory syncytial virus",
    "# 20 Rotavirus",
    "# 21 Rubella",
    "# 22 Tetanus",
    "# 23 Tuberculosis",
    "# 24 Varicella (chickenpox)",
    "# 25 Zoster (herpes zoster/shingles)",
    "# Appendix 1: The history of immunisation in New Zealand",
    "# Appendix 2: Planning immunisation catch-ups",
    "# Appendix 3: Immunisation standards for vaccinators and guidelines for organisations offering immunisation services",
    "# Appendix 4: Authorisation and criteria of vaccinators",
    "# Appendix 5: Immunisation certificate",
    "# Appendix 6: Passive immunisation",
    "# Appendix 7: Vaccine presentation, preparation, disposal, and needle-stick recommendations",
    "# Appendix 8: Websites and other online resources",
    "# Appendix 9: Medicines approved for use in immunisation programmes",
    "# Commonly used abbreviations",
    "# Glossary of vaccine brand names and abbreviations",
    "END_OF_FILE" # 虛擬終點
]

def build_index_map(chapters_list, full_text):
    index_map = {}
    current_search_pos = 0
    for ch in chapters:
        if ch == "END_OF_FILE":
            index_map[ch] = len(full_text)
        else:
            pos = full_text.find(ch, current_search_pos)
            if pos != -1:
                index_map[ch] = pos
                current_search_pos = pos + len(ch)
            else:
                fallback_pos = full_text.find(ch)
                if fallback_pos != -1:
                    index_map[ch] = fallback_pos
                    current_search_pos = fallback_pos + len(ch) 
                else:
                    print(f"嚴重錯誤：找不到章節標題 [{ch}]，請檢查 MD 內容格式！該章節將被跳過。")
                    continue
    return index_map

my_index_map = build_index_map(chapters, full_text)
#print(my_index_map)

def get_transcript_content(parent_folder, case_id):
    """
    根據 Parent_Folder 和 Case_ID 構建路徑並讀取 JSON
    """
    # 構建完整路徑，例如：.../output20260417/d6c68dff...json
    file_path = os.path.join(BASE_TRANSCRIPT_PATH, parent_folder, case_id)
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取對話內容
                transcript_text = ""
                if 'Transcript' in data:
                    for item in data['Transcript']:
                        role = item.get('ParticipantId', 'Unknown')
                        content = item.get('Content', '')
                        transcript_text += f"{role}: {content}\n"
                
                #if isinstance(transcript, list):
                #    # 如果 transcript 是列表格式，將其轉換為帶換行的字串
                #    return "\n".join([f"{item.get('speaker', 'UNKNOWN')}: {item.get('text', '')}" for item in transcript])
                return transcript_text
        except Exception as e:
            return f"Error reading JSON: {e}"
    else:
        return f"File Not Found: {file_path}"

def get_ai_audit(query, manual_context):
    print(f"DEBUG: Manual context length: {len(manual_context)}")
    #print(f"DEBUG: Sample of context: {manual_context[:200]}")

    version = "old"
    NEW_AUDIT_TEMPLATE = """
    [CONFIRMATION]
    I, the AI Auditor, acknowledge that the User has provided a [Reference Material] section below. 
    I MUST parse this material before answering. 
    I will NOT state that the handbook is unavailable.

    # Role
    You are the "Authoritative Audit Expert" for the New Zealand Immunisation Handbook 2026. 
    Your mission is to generate a "Golden Truth" answer based on a consultation transcript to serve as a benchmark for evaluating other AI agents.

    # Reference Material (Markdown Context)
    [MANUAL_CONTEXT]

    # Transcript to Analyze (Query)
    [USER_QUERY]

    # Task Instructions
    1. **Identify Clinical Facts**: Correctly identify vaccine types, clinical context, and any "vaccination errors" (e.g., expired vaccines, wrong intervals) mentioned in the transcript.
    2. **Clinical Rules (Technical)**: Refer EXCLUSIVELY to the provided [Reference Material] (especially vaccine chapters and Appendix 2) to judge the technical correctness of the advice.
    3. **Professional Standards (Procedural)**: Use **Appendix 3 (Immunisation standards)** to evaluate the professional conduct. Assess if the healthcare provider followed protocols for:
        - Reporting and managing vaccination errors (Standard 3).
        - Clinical governance and seeking expert advice when uncertain (Standard 1).
    4. **Safety Protocol**: If the manual is silent on a specific error, follow the most conservative protocol: document the error, cite the violation of professional standards, and recommend expert consultation.

    # Output Requirements (Strict JSON Format)
    You MUST respond in JSON format. Do not include any text before or after the JSON.

    {
        "ground_truth": "[Provide full clinical and professional advice here, following the structure below:
                        [SOURCE_CHAPTERS]: List all relevant chapters including Appendix 3 if applicable.
                        [CLINICAL_ADVICE]: Detailed judgment on the vaccine schedule/dose.
                        [PROFESSIONAL_CONDUCT_AUDIT]: Evaluate if the error handling aligns with Appendix 3 standards.
                        [ACTION_STEPS]: Concrete next steps for the provider.
                        [SAFETY_WARNING]: Standard medical disclaimer.
                        [CLASSIFICATION_TAGS]: Metadata info.]",
        "context": "[Provide ONLY the ORIGINAL_ENGLISH_QUOTES here, formatted as: From Chapter X: 'Exact quote']"
    }

    # Output Language
    Respond entirely in ENGLISH.
    """

    OLD_AUDIT_TEMPLATE = """
        # Role
        You are the "Authoritative Audit Expert" for the New Zealand Immunisation Handbook 2026. 
        Your mission is to generate a "Golden Truth" answer based on a consultation transcript to serve as a benchmark for evaluating other AI agents.
        Your goal is to produce a flawless, highly detailed audit response.

        # Task Instructions
        1. Identify correctly the vaccine types and clinical context from the transcript.
        2. Refer EXCLUSIVELY to the provided [Reference Material] for clinical rules. 
        3. If the transcript involves complex catch-up schedules, strictly align the patient's age and previous doses with the tables in the handbook.
        4. If the manual is silent on a specific error (e.g., a 7-week gap), follow the most conservative safety protocol: document the error and recommend expert consultation.
        5. Step-by-Step Reasoning (Internal Thought):
           Before finalizing the JSON, internally verify the following:
                - Compare the patient's age in the [Transcript] against the 'Minimum Age' column in the [Reference Material].
                - Calculate the interval between doses carefully.
                - Verify if the specific vaccine brand mentioned has specific contraindications in the handbook.
        6. If the [Reference Material] does not contain the answer, you MUST state 'Reference material insufficient' instead of using external medical knowledge.

        # Output Requirements (Strict JSON Format)
        You MUST respond in JSON format with the following keys, so the script can parse it:

        {
            "ground_truth": "(Provide full clinical advice. This string MUST include the exact headers below. Use '\n' for each new line to ensure readability):
                [SOURCE_CHAPTERS]: (List exact Chapter and Section numbers with Titles)
                [CLINICAL_ADVICE]: (Provide a professional answer. If no info found, state: 'The provided guidance is silent on this specific query.')
                [KEY_EVIDENCE]: (Provide EXACT verbatim quotes ONLY from Reference Material. You must extract 2-3 core sentences that directly justify the advice. Do not paraphrase, do not translate, and do not modify the original wording. If no direct evidence is found, state: 'No verbatim evidence available.')
                [TRANSPARENCY_INDICATOR]:
                    [Source Confidence]: (Select one: High/Medium/Low)
                    [Provenance]: (e.g., Main text, Funding Table, or Appendix)
                    [Status]: (Select one: Found / Partially Found / Not Found in Handbook)
                [SAFETY_WARNING]: (This information is for clinical guidance only. Final decisions rest with the healthcare professional.)
                [CLASSIFICATION_TAGS]:
                    [Vaccine Type]: (e.g., COVID-19, HPV, MMR)
                    [Clinical Scenario]: (e.g., Funding, Catch-up, Administration Error)
                    [Urgency]: (Routine / High)"
            "context": "(You MUST extract literal paragraphs from Reference Material. If multiple sections are used, provide all relevant quotes.
                - MANDATORY FORMAT: Each quote must start with this exact prefix: [From Chapter X Section Y]: (Replace X and Y with actual numbers)
                - LINE BREAKS: Use a literal '\n' after each quote segment to ensure each prefix starts on a new line.
                - EXAMPLE: [From Chapter 4 Section 4.2]: 'The vaccine should be stored at...' \n [From Chapter 9 Section 9.5]: 'Patients with...'
                - CRITICAL: Do not summarize. Do not include these instructions in the output.
        }

        # Reference Material (Markdown Context)
        [MANUAL_CONTEXT]

        # Transcript to Analyze (Query)
        [USER_QUERY]

        # Output Language
        Respond entirely in ENGLISH.
    """

    if version == "new":
        prompt = NEW_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", manual_context).replace("[USER_QUERY]", query)
    else:
        prompt = OLD_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", manual_context).replace("[USER_QUERY]", query)

    final_messages = [{"role": "user", "content": prompt}]
    # 精確計算「即將發出」的總 Token 數
    enc = tiktoken.encoding_for_model(deployment_name)
    total_tokens = 0
    for msg in final_messages:
        total_tokens += len(enc.encode(msg["content"]))
    print(f"最終發送總 Token 數: {total_tokens}")

    #try:
    response = client.chat.completions.create(
        model=deployment_name, 
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" },
        reasoning_effort="medium"
    )
    return json.loads(response.choices[0].message.content), total_tokens
    #except Exception as e:
    #    return {"ground_truth": f"Error: {e}", "context": "Error"}

# --- 執行測試 ---
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def librarian_agent_with_retry(text, chapters):
    return librarian_agent(text, chapters)

#@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_fixed(5),
    before_sleep=lambda retry_state: print(f"--- 第 {retry_state.attempt_number} 次重試 --- 錯誤原因: {retry_state.outcome.exception()}")
)
def get_ai_audit_with_retry(query, manual_context):
    return get_ai_audit(query, manual_context)

# Step 1: 讓 AI 選章節
# 讀取輸入檔
df = pd.read_csv(input_file)
df = df.dropna(subset=['Case_ID', 'Parent_Folder'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
if 'ground_truth' not in df.columns:
    df['ground_truth'] = ""
if 'context' not in df.columns:
    df['context'] = ""
if 'chosen_chapters' not in df.columns:
    df['chosen_chapters'] = ""
if 'length_of_ref' not in df.columns:
    df['length_of_ref'] = ""
if 'ref_st' not in df.columns:
    df['ref_st'] = ""
if 'ref_end' not in df.columns:
    df['ref_end'] = ""
if 'num_tokens' not in df.columns:
    df['num_tokens'] = ""

for index, row in df.iterrows():
    # 從 CSV 抓取對應的目錄與檔名
    folder = str(row['Parent_Folder'])
    file_name = str(row['Case_ID'])    
    print(f"正在讀取: {folder}/{file_name}")

    # A. 取得對話原文並存入 query 欄位
    transcript_text = get_transcript_content(folder, file_name)
    df.at[index, 'query'] = transcript_text
    try:
        result = librarian_agent_with_retry(transcript_text, chapters)
    except Exception:
        print("librarian_agent 重試 5 次後依然失敗")
        break

    print(f"   AI 選擇了: {result}")

    if isinstance(result, list):
        chosen_chapters = result
    else:
        # 萬一 AI 耍笨只回傳了一個字串，強迫轉成 List
        chosen_chapters = [result]
    #chosen_chapters = ["# Appendix 2: Planning immunisation catch-ups", "# 4 Immunisation of special groups", "# 5 Coronavirus disease (COVID-19)"]
    # Step 2: 擷取對應文字
    reference_text = get_combined_context(chosen_chapters, my_index_map, full_text, chapters)
    #print(reference_text)
    # Step 3: 送去給 Auditor 做最終判斷
    try:
        audit_result, total_tokens = get_ai_audit_with_retry(transcript_text, reference_text)
    except Exception:
        print("get_ai_audit 重試 5 次後依然失敗")
        break
    #audit_result = get_ai_audit(transcript_text, reference_text)
    agent_output = audit_result.get("ground_truth") or audit_result.get("error") or "No Response Found"
    agent_context = audit_result.get("context") or "N/A"

    # Step 4: 儲存結果 (存成 txt 或 md)
    df.at[index, 'ground_truth'] = agent_output
    df.at[index, 'context'] = agent_context
    df.at[index, 'chosen_chapters'] = chosen_chapters
    df.at[index, 'length_of_ref'] = len(reference_text)
    df.at[index, 'ref_st'] = reference_text[:500] + "..."
    df.at[index, 'ref_end'] = "..." + reference_text[-500:]
    df.at[index, 'num_tokens'] = total_tokens
    
    # 檢查檔案是否已存在，決定是否要寫入欄位名稱 (Header)
    file_exists = os.path.isfile(output_file)
    current_row_df = df.iloc[[index]]
    # 使用 mode='a' (append) 模式續寫
    current_row_df.to_csv(output_file, mode='a', index=False, header=not file_exists, encoding='utf-8-sig', escapechar='\\')

    print(f"進度: {index+1}/{len(df)} - {row['Topic']}")

    # 🛠️ 加在 For 迴圈的最後一行
    time.sleep(2) # 👈 強制冷卻，隔絕批次執行時伺服器端的 Caching 記憶交織

#df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"成功！輸出：{output_file}")
end_time = time.time()    # 紀錄結束時間
duration = end_time - start_time

print(f"執行時間: {duration:.2f} 秒")
