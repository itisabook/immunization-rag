import json
from openai import OpenAI  # 注意：這裡是 OpenAI 而不是 AzureOpenAI
import pandas as pd
import os
import datetime
import time
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")

start_time = time.time()  # 紀錄開始時間
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 從你剛才的 Sample Code 複製過來的資訊
endpoint = "https://ai-team-02-hack.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5-mini"  # librarian
deployment_name_2 = "gpt-5-mini"  # related context extractor
deployment_name_3 = "gpt-5-mini"  # ai-audit

BASE_TRANSCRIPT_PATH = r"c:\Users\Owner\Documents\compsci714\hackathon\immunization_guidelines\output20260400\local-folder"
input_file = "test_benchmark_25_20260522_025026_new.csv"
#"test_benchmark_100_20260530_003852.csv"
#"test_benchmark_25_20260522_041352_new.csv"



md_file = "NZ_Immunisation_Handbook_2026_by_marker.md"
#"immunisation-handbook-2026-v2.md"
output_file = f"results_{Path(input_file).stem}_{timestamp}.csv"

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
            Select this parent chapter (# 2) ONLY IF the query covers the entire clinical process.
            Note: If you select a specific subsection below (e.g., ## 2.2), DO NOT include the parent chapter (# 2).
            - ## 2.1 Pre-vaccination (Clinical assessment, Screening, Contraindications, Precautions, Co-administration of vaccines, Cold Chain concerns, Fever/Acute illness at the time of vaccination, Informed consent)
            - ## 2.2 Vaccine administration (Injection techniques: Intramuscular/Subcutaneous/Intradermal, Needle size/length, Anatomical sites: Deltoid / Vastus lateralis / Ventrogluteal, Positioning, Multiple injections)
            - ## 2.3 Post-vaccination (Observation period, Management of Anaphylaxis, Fainting/Syncope, Common side effects, Post-vaccination advice/education)
        - # 3 Vaccination questions and addressing concerns：
            涉及疫苗時程安排、嬰幼兒接種問題、過敏與疾病者是否能打針的判斷、活毒疫苗的傳播風險、和疫苗接種的常見誤解與擔憂。
        - # 4 Immunisation of special groups: 
            Select this parent chapter (# 4) ONLY IF the query covers multiple special groups.
            Note: If you select a specific subsection below (e.g., ## 4.3), DO NOT include the parent chapter (# 4). Always prioritize the most granular level to minimize context tokens.
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
    api_kwargs = {
        "model": deployment_name,
        "messages": [{"role": "system", "content": prompt},
                  {"role": "user", "content": transcript}],
        "response_format": {"type": "json_object"}
    }
    if "gpt-4o-mini" in deployment_name:
        api_kwargs["temperature"] = 0
        #api_kwargs["reasoning_effort"]="medium"
        #api_kwargs["reasoning"]="medium"

    response = client.chat.completions.create(**api_kwargs)
    total_tokens = response.usage.total_tokens if response.usage else 0
    print(f"Librarian Actual Tokens: {total_tokens}")

    return json.loads(response.choices[0].message.content)["selected_chapters"], total_tokens

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
        current_idx = chapters_list.index(key)
                
        # 判斷當前標題層級
        # 如果是以 "# " 開頭 (大章)，如果是 "## " 開頭 (小節)
        is_level_1 = key.strip().startswith("# ") 

        end = len(full_text) # 預設為文件結尾

        # 尋找下一個終止點
        for i in range(current_idx + 1, len(chapters_list)):
            next_key = chapters_list[i]
            
            if is_level_1:
                # [大章模式]：尋找下一個 "# " 開頭的標題 (例如從 # 4 找下一個 # 5)
                # 忽略中間所有 "## " 開頭的小節
                if next_key.strip().startswith("# "):
                    end = index_map[next_key]
                    break
            else:
                # [小節模式]：遇到任何下一個標題 (無論是 # 或 ##) 就停止
                # 例如從 ## 4.1 找到 ## 4.2 停止
                end = index_map[next_key]
                break

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
    # --- 第二章細化開始 ---
    "# 2 Processes for safe immunisation",
    "## 2.1 Pre-vaccination",
    "## 2.2 Vaccine administration",
    "## 2.3 Post-vaccination",
    # --- 第二章細化結束 ---
    "# 3 Vaccination questions and addressing concerns",
    # --- 第四章細化開始 ---
    "# 4 Immunisation of special groups",
    "## 4.1 Pregnancy and lactation",
    "## 4.2 Infants with special immunisation considerations from birth",
    "## 4.3 Immunocompromised individuals",
    "## 4.4 Chronic kidney disease",
    "## 4.5 Chronic liver disease",
    "## 4.6 Other special groups",
    "## 4.7 Immigrants and refugees",
    "## 4.8 Occupation-related vaccination",
    "## 4.9 Travel",
    # --- 第四章細化結束 ---
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

def get_ai_evidence(query, manual_context):
    print(f"DEBUG (get_ai_evidence): Manual context length: {len(manual_context)}")
    version = "0605"
    NEW_0605_AUDIT_TEMPLATE = """
        # Role: 你是高階醫療文獻提煉官。你的任務是從龐大的 (Reference Material) 中，找出能解答使用者問題（User Query）的核心章節。

        # Execution Rules:
            1. 結構完整原則（CRITICAL）：一旦在文獻中定位到與問題相關的臨床答案，你必須提取該答案所在的【整個次級章節（ENTIRE sub-section）全文】。
            2. 提取範圍定義：必須包含該段落上方的次級標題（例如：## 2.x 或 ### 2.x.x）、該標題底下的所有連續段落、所有的項目列表 (Bullet points)、以及緊隨其後的備註說明或警告區塊 (Notes/Cautions)。
            3. 絕對不可只斷章取義地提取單一句子或單一數據。你的核心任務是把「包含答案的那一整塊完整上下文（Full block of context）」打包帶走，好讓下一棒的 Agent 擁有宏觀的視野來回答複雜的前線行政與臨床問題。
            4. 提取出來的內容必須完全保持原文的字眼（Verbatim），不要進行任何摘要、潤飾或縮寫。
            5. 如果你翻遍了整個文獻，發現裡面連一個相關的疫苗字眼或章節都完全沒有提及，請直接輸出：No related context available.

        # Output Format:
        直接輸出篩選後的純文字段落（包含你提取的章節標題與全文）。絕對不要包含任何 JSON 語法、不要包在 ```markdown 或 ```text 代碼塊中。

        # Reference Material (Manual Context):
        [MANUAL_CONTEXT]

        # User Query:
        [USER_QUERY]
    """
    NEW_AUDIT_TEMPLATE = """
        # Role: 你是嚴格的醫療文獻篩選官。你的任務是從龐大的 Reference Material 中，找出所有與使用者問題User Query「直接相關」或「沾到邊」的完整段落或表格。

        # Execution Rules:
        1. 提取出來的內容必須完全保持原文的字眼（Verbatim），不要進行任何摘要、潤飾或翻譯。
        2. 請把所有相關的段落完整保留，組合成一段乾淨的 related context 輸出。
        3. 你的目標是進行資訊減速與純化。將不相關的章節段落徹底丟棄。
        4. 如果你翻遍了整個 Reference Material，發現裡面完全沒有任何內容與使用者的問題相關，你必須「只」輸出這行字，絕對不要多寫：No related context available.

        # Output Format:
        直接輸出篩選後的純文字段落。絕對不要包含任何 JSON 語法、不要包在 ```markdown 或 ```text 代碼塊中。

        # Reference Material (Manual Context):
        [MANUAL_CONTEXT]

        # User Query:
        [USER_QUERY]
    """

    OLD_AUDIT_TEMPLATE = """
        # Role: 你是熱心且細心的醫療文獻篩選官。你的任務是從龐大的 (Reference Material) 中，找出所有與使用者問題（User Query）「直接相關」、「間接相關」或「可能沾到邊（包含同義詞、疾病大分類、疫苗一般禁忌症）」的完整段落或表格。

        # Execution Rules:
        1. 提取出來的內容必須完全保持原文的字眼（Verbatim），不要進行任何摘要、潤飾或翻譯。
        2. 請把所有相關的段落完整保留，組合成一段乾淨的 related context 輸出。
        3. 寬鬆檢索原則：只要手冊內容提及了與問題相關的疫苗、相關的免疫系統疾病、補助表格、或是類似症狀的處理流程，都算「沾到邊」，請務必保留，寧可多撈，絕漏掉任何潛在線索！
        4. 如果你翻遍了整個 (Reference Material)，發現裡面「連一個相關的疫苗字眼、章節或表格都完全沒有提及」，你才輸出這行字：No related context available.

        # Output Format:
        直接輸出篩選後的純文字段落。絕對不要包含任何 JSON 語法、不要包在 ```markdown 或 ```text 代碼塊中。

        # Reference Material (Manual Context):
        [MANUAL_CONTEXT]

        # User Query:
        [USER_QUERY]
    """

    if version == "new":
        prompt = NEW_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", manual_context).replace("[USER_QUERY]", query)
    elif version == "old":
        prompt = OLD_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", manual_context).replace("[USER_QUERY]", query)
    else:
        prompt = NEW_0605_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", manual_context).replace("[USER_QUERY]", query)        

    final_messages = [{"role": "user", "content": prompt}]
    # 精確計算「即將發出」的總 Token 數
    enc = tiktoken.encoding_for_model(deployment_name_2)
    total_sent_tokens = 0
    for msg in final_messages:
        total_sent_tokens += len(enc.encode(msg["content"]))

    api_kwargs = {
        "model": deployment_name_2,
        "messages": [{"role": "user", "content": prompt}],
        #"response_format": {"type": "json_object"}
    }
    if "gpt-4o-mini" in deployment_name_2:
        api_kwargs["temperature"] = 0
        #reasoning_effort="medium"

    response = client.chat.completions.create(**api_kwargs)
    
    # 4. 取得 AI 回傳的純文字內容與 Token
    raw_content = response.choices[0].message.content.strip()
    #print(raw_content)
    total_tokens = response.usage.total_tokens if response.usage else 0
    print(f"Estimate SENT Tokens: {total_sent_tokens}, Actual Tokens: {total_tokens}")

    # 直接回傳純文字字串，這樣在主程式中用 if "No verbatim evidence" in raw_content 判斷極度乾淨穩定！
    return raw_content, total_tokens


def get_ai_audit(query, related_context):
    print(f"DEBUG (get_ai_audit): context_evidence length: {len(related_context)}")
    #print(related_context[:500])
    version = "new"
    NEW_AUDIT_TEMPLATE = """
        # Role: 你是 IMAC (Immunisation Advisory Centre) 的臨床顧問 AI 助手。你的任務是根據提供之已過濾的相關手冊內容 (Reference Material)，為醫護人員提供精確、合規的臨床建議。

        # Language Protocol:
        Respond entirely in ENGLISH. DO NOT TRANSLATE ANY PART OF THE CLINICAL RESPONSE.

        # Knowledge Handling & Transparency:
        1. 你的所有臨床建議，必須「唯一且完全（ONLY and ENTIRELY）」衍生自 (Reference Material) 內的宣稱，絕對不允許使用你大腦中原有的任何醫學常識。
        2. 絕對不要延伸、不要腦補、不要假設。

        # Output Requirements (Strict JSON Format):
        You must return a valid JSON object. Do NOT wrap the response in ```json ``` codeblocks. Do not include any greeting, preamble, or postscript text before or after the JSON object.

        JSON Structure:
        {
            "agent_context": "[PLACEHOLDER_FOR_PYTHON_CONTEXT]",
            "agent_response": "[SOURCE_CHAPTERS]: (List the exact Chapter and Section numbers with Titles found in the context)
            [KEY_EVIDENCE]: (Extract 2-3 core sentences VERBATIM (word-for-word) ONLY from the provided (Reference Material) that directly support the advice. Do not paraphrase, do not translate, and do not modify the original wording. If no direct evidence is found, state exactly: 'No verbatim evidence available.')
            [CLINICAL_ADVICE]: (Provide a professional, clear answer to the user's query. Your advice MUST be derived ONLY and ENTIRELY from the facts stated in the [KEY_EVIDENCE] section above. If the [KEY_EVIDENCE] is 'No verbatim evidence available', you MUST state exactly: 'The provided guidance is silent on this specific query.')
            [TRANSPARENCY_INDICATOR]
                [Source Confidence]: (Select one: High/Medium/Low)
                [Provenance]: (Select one: Main text, Funding Table, or Appendix)
                [Status]: (Select one: Found / Partially Found / Not Found in Handbook)
            [SAFETY_WARNING]: This information is for clinical guidance only. Final decisions rest with the healthcare professional.
            [CLASSIFICATION_TAGS]
                [Vaccine Type]: (e.g., COVID-19, HPV, MMR)
                [Clinical Scenario]: (e.g., Funding, Catch-up, Administration Error)
                [Urgency]: (Routine / High)"
        }

        === OUTPUT CONSTRAINT (CRITICAL) ===
        1. MANDATORY TAGS & ORDER: Every bracketed tag must appear inside the 'agent_response' string in this exact chronological order:
        [SOURCE_CHAPTERS] -> [KEY_EVIDENCE] -> [CLINICAL_ADVICE] -> [TRANSPARENCY_INDICATOR] -> [SAFETY_WARNING] -> [CLASSIFICATION_TAGS]
        2. QUOTE HANDLING: Use ONLY single quotes (') for any text, vaccine names, or quotation marks inside the 'agent_response' string value. DOUBLE QUOTES (") are STRICTLY FORBIDDEN inside the string text because they break the JSON structure.
        3. NO WRAPPER: Do not wrap the JSON object in ```json ... ``` code blocks. Output ONLY the raw JSON object.
        4. LINE BREAKS: All conceptual section breaks inside the 'agent_response' string must be represented as a literal '\n\n' or '\n' characters.

        # Reference Material (Markdown Context)
        [MANUAL_CONTEXT]

        # Transcript to Analyze (Query)
        [USER_QUERY]
    """

    OLD_AUDIT_TEMPLATE = """
        # Role: 你是 IMAC (Immunisation Advisory Centre) 的臨床顧問 AI 助手。你的目標是根據提供之《紐西蘭免疫手冊》(Immunisation Handbook) 檢索內容，為醫護人員提供精確、具證據支持的臨床建議。

        # Language Protocol:
            Respond entirely in ENGLISH. DO NOT TRANSLATE ANY PART OF THE CLINICAL RESPONSE.

        # Knowledge Handling & Transparency (核心邏輯):
            - ## Source-Only Principle: 
                Only provide advice found in the provided reference material.
            - ## The "Clear Not Found" Rule: 
                If the provided text does not contain the specific answer (e.g., a specific age limit or funding criteria), you MUST explicitly state: "The provided official guidance is silent on this specific query." After stating this, you may provide a conservative clinical safety action (e.g., consult an IMAC specialist).
            - ## Confidence Assessment: 
                Evaluate if the information is explicitly stated (High Confidence) or if it requires clinical inference (Medium/Low Confidence).
            - ## Chapter Association: 
                Automatically associate vaccines with their respective chapters (e.g., Hepatitis B = Chapter 9) even if numbers are not explicitly in the snippet.
            - ## Ambiguity Detection: 
                Always evaluate if the guidance is explicit. If the manual is silent or provides conflicting information, prioritize patient safety by flagging the uncertainty rather than making assumptions.

        # Output Requirements (Strict JSON Format):
            You must return a valid JSON object. Your response is strictly for an automated audit system, so formatting is as important as content.

            JSON Structure:
            {
                "agent_context": "(You MUST extract literal paragraphs from Reference Material.
                    - MANDATORY FORMAT: Each quote must start with this exact prefix: [From Chapter X Section Y]: (Replace X and Y with actual numbers)
                    - LINE BREAKS: Use a literal '\n' after each quote segment to ensure each prefix starts on a new line.
                    - EXAMPLE: [From Chapter 4 Section 4.2]: 'The vaccine should be stored at...' \n [From Chapter 9 Section 9.5]: 'Patients with...'
                    - WORD COUNT: Aim for at least 250 words in total. If the source is shorter, copy all relevant text.
                    - CRITICAL: Do not summarize. Do not include these instructions in the output.)"
                "agent_response": "(Provide full clinical advice. This string MUST include the exact headers below. Use '\n' for each new line to ensure readability):
                    [SOURCE_CHAPTERS]: (List exact Chapter and Section numbers with Titles)
                    [CLINICAL_ADVICE]: (Provide a professional answer. Your answer must come from "agent_context". If no info found, state: 'The provided guidance is silent on this specific query.')
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
            }

        === OUTPUT CONSTRAINT (CRITICAL) ===
        1. MANDATORY TAGS: Every bracketed tag (e.g., [SOURCE_CHAPTERS]) must appear in the 'agent_response' string. 
        2. QUOTE HANDLING: Use ONLY single quotes (') for text inside the JSON values. DOUBLE QUOTES (") are strictly forbidden except for the JSON keys themselves.
        3. NO WRAPPER: Do not include ```json blocks or any text before/after the JSON object.
        4. ESCAPING: Ensure all line breaks are represented as '\n' inside the JSON string.

        # Reference Material (Markdown Context)
        [MANUAL_CONTEXT]

        # Transcript to Analyze (Query)
        [USER_QUERY]
    """

    if version == "new":
        prompt = NEW_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", related_context).replace("[USER_QUERY]", query)
    else:
        prompt = OLD_AUDIT_TEMPLATE.replace("[MANUAL_CONTEXT]", related_context).replace("[USER_QUERY]", query)

    final_messages = [{"role": "user", "content": prompt}]
    # 精確計算「即將發出」的總 Token 數
    enc = tiktoken.encoding_for_model(deployment_name_3)
    total_sent_tokens = 0
    for msg in final_messages:
        total_sent_tokens += len(enc.encode(msg["content"]))
    
    api_kwargs = {
        "model": deployment_name_3,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    if "gpt-4o-mini" in deployment_name_3:
        api_kwargs["temperature"] = 0
        #reasoning_effort="medium"

    response = client.chat.completions.create(**api_kwargs)
    total_tokens = response.usage.total_tokens if response.usage else 0
    print(f"Estimate SENT Tokens: {total_sent_tokens}, Actual Tokens: {total_tokens}")

    try:
        json_data = json.loads(response.choices[0].message.content)
        json_data["agent_context"] = related_context
        return json_data, total_tokens
        
    except json.JSONDecodeError:
        # 防禦性 Error Handling
        print("ERROR: Step 3 輸出 JSON 解析出錯，啟動 Python 安全墊背回覆。")
        return {
            "agent_context": related_context,
            "agent_response": f"[CLINICAL_ADVICE]\nRefer to the related context above."
        }, 0


# --- 執行測試 ---
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def librarian_agent_with_retry(text, chapters):
    return librarian_agent(text, chapters)

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_fixed(5),
    before_sleep=lambda retry_state: print(f"--- 第 {retry_state.attempt_number} 次重試 --- 錯誤原因: {retry_state.outcome.exception()}")
)
def get_ai_evidence_with_retry(query, manual_context):
    return get_ai_evidence(query, manual_context)

#@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
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
if 'agent_response' not in df.columns:
    df['agent_response'] = ""
if 'agent_context' not in df.columns:
    df['agent_context'] = ""
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
if 'raw_dbg' not in df.columns:
    df['raw_dbg'] = ""

for index, row in df.iterrows():
    # 從 CSV 抓取對應的目錄與檔名
    folder = str(row['Parent_Folder'])
    file_name = str(row['Case_ID'])    
    print(f"正在讀取: {folder}/{file_name}")

    # A. 取得對話原文並存入 query 欄位
    transcript_text = get_transcript_content(folder, file_name)
    df.at[index, 'query'] = transcript_text
    try:
        result, token_librarian = librarian_agent_with_retry(transcript_text, chapters)
    except Exception:
        print("librarian_agent 重試 5 次後依然失敗")
        break

    print(f"   AI 選擇了: {result}")

    if isinstance(result, list):
        chosen_chapters = result
    else:
        # 萬一 AI 耍笨只回傳了一個字串，強迫轉成 List
        chosen_chapters = [result]

    # 過濾邏輯：如果清單中同時存在 "## X.Y" 和 "# X"，則移除 "# X"
    refined_sections = []
    for section in chosen_chapters:
        print(section)
        # 檢查是否為大章節 (例如 # 4)
        if section.startswith('# ') and not section.startswith('##'):
            chapter_num = section.split(' ')[1] # 取得 "4"
            # 檢查清單中是否已經有對應的子章節 (例如 ## 4.3)
            has_child = any(s.startswith(f'## {chapter_num}.') for s in chosen_chapters)
            if not has_child:
                refined_sections.append(section)
        else:
            refined_sections.append(section)

    # Step 2: 擷取對應文字
    reference_text = get_combined_context(refined_sections, my_index_map, full_text, chapters)
    enc = tiktoken.encoding_for_model(deployment_name_2)
    tokens = len(enc.encode(reference_text))
    print(f"reference_text 實際用 Token 數: {tokens}")
    tokens = len(enc.encode(transcript_text))
    print(f"transcript_text 實際用 Token 數: {tokens}")

    # Step 3: 送去給 Evidence Extractor 做最終判斷
    try:
        evidence_result, token_evidence = get_ai_evidence_with_retry(transcript_text, reference_text)
    except Exception:
        print("get_ai_evidence 重試 5 次後依然失敗")
        break

    # Step 4: 送去給 Auditor 做最終判斷
    try:
        audit_result, token_audit = get_ai_audit_with_retry(transcript_text, evidence_result)
    except Exception:
        print("get_ai_audit 重試 5 次後依然失敗")
        break
    agent_output = audit_result.get("agent_response") or audit_result.get("error") or "No Response Found"
    agent_context = audit_result.get("agent_context") or "N/A"

    # Step 4: 儲存結果 (存成 txt 或 md)
    df.at[index, 'agent_response'] = agent_output
    df.at[index, 'agent_context'] = agent_context
    df.at[index, 'chosen_chapters'] = refined_sections
    df.at[index, 'length_of_ref'] = len(reference_text)
    df.at[index, 'ref_st'] = reference_text[:500] + "..."
    df.at[index, 'ref_end'] = "..." + reference_text[-500:]
    df.at[index, 'num_tokens'] = token_librarian + token_evidence + token_audit
    df.at[index, 'raw_dbg'] = str(audit_result)
    
    # 檢查檔案是否已存在，決定是否要寫入欄位名稱 (Header)
    file_exists = os.path.isfile(output_file)
    current_row_df = df.iloc[[index]]
    # 使用 mode='a' (append) 模式續寫
    current_row_df.to_csv(output_file, mode='a', index=False, header=not file_exists, encoding='utf-8-sig', escapechar='\\')

    print(f"進度: {index+1}/{len(df)} - {row['Topic']}")

print(f"成功！輸出：{output_file}")
end_time = time.time()    # 紀錄結束時間
duration = end_time - start_time

print(f"執行時間: {duration:.2f} 秒")
