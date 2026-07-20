import asyncio
#from openai import OpenAI  # 注意：這裡是 OpenAI 而不是 AzureOpenAI

# =====================================================================
# 🌟 大魔王強制攔截補丁（Monkey Patch）
# 必須放在所有 import azure 程式碼的最前面！
# =====================================================================
import openai
import io
from contextlib import redirect_stdout
import json
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")


# 攔截異步請求 (SDK底層主要用這個)
original_async_create = openai.resources.chat.completions.AsyncCompletions.create
async def patched_async_create(self, *args, **kwargs):
    # 1. 處理 max_tokens
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    
    # 2. 處理 temperature (強制拔掉，讓模型回歸預設值 1)
    if "temperature" in kwargs:
        kwargs.pop("temperature")
    return await original_async_create(self, *args, **kwargs)
openai.resources.chat.completions.AsyncCompletions.create = patched_async_create

# 攔截同步請求 (保險起見一起攔截)
original_sync_create = openai.resources.chat.completions.Completions.create
def patched_sync_create(self, *args, **kwargs):
    # 1. 處理 max_tokens
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    
    # 2. 處理 temperature (強制拔掉，讓模型回歸預設值 1)
    if "temperature" in kwargs:
        kwargs.pop("temperature")
    return original_sync_create(self, *args, **kwargs)
openai.resources.chat.completions.Completions.create = patched_sync_create
# =====================================================================

from azure.ai.evaluation import RelevanceEvaluator, GroundednessEvaluator, RetrievalEvaluator
from azure.identity import DefaultAzureCredential
model_config = {
    "azure_endpoint": "https://ai-team-02-hack.services.ai.azure.com",
    "azure_deployment": "gpt-5-mini",                            # 你的模型部署名稱（建議用 GPT-4o 或 GPT-4）
    "api_version": "2024-06-01",                              # API 版本
    #"api_version": "2024-02-15-preview",
    "api_key": my_api_key
}

# Initialize an evaluator:
credential = DefaultAzureCredential()
relevance_eval = RelevanceEvaluator(model_config, credential=credential, threshold=4)
groundedness_eval = GroundednessEvaluator(model_config, credential=credential, threshold=4)
retrieval_eval = RetrievalEvaluator(model_config, credential=credential, threshold=4)

#query = "What is the capital of life?"
#response = "Paris."
#relevance_score = relevance_eval(query=query, response=response)
#print(relevance_score["relevance_score"])
#
#conversation = {
#    "messages": [
#        { "content": "Which tent is the most waterproof?", "role": "user" },
#        { "content": "The Alpine Explorer Tent is the most waterproof", "role": "assistant", "context": "From the our product list the alpine explorer tent is the most waterproof. The Adventure Dining Table has higher weight." },
#        { "content": "How much does it cost?", "role": "user" },
#        { "content": "$120.", "role": "assistant", "context": "The Alpine Explorer Tent is $120."}
#    ]
#}
#groundedness_conv_score = groundedness_eval(conversation=conversation)
#print(groundedness_conv_score["groundedness_score"])

input_json_path = "myevalresults.json"
GLOBAL_METRICS_JSON_PATH = "final_global_metrics.json"
output_jsonl_path = "myevalresults_output.jsonl"
DATASET_PATH = "ai_foundry_eval_dataset.jsonl"
BACKUP_PATH = "ai_foundry_eval_dataset_backup.jsonl"

from azure.ai.evaluation import evaluate

if not os.path.exists(BACKUP_PATH):
    shutil.copy(DATASET_PATH, BACKUP_PATH)
with open(BACKUP_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
current_round_global_indices = list(range(total_lines))  # 初始：[0, 1, 2, ..., 10]

attempt = 0
attempt_limit = 2
current_all_rows = []
while True:
    attempt += 1
    print(f"\n🔄 [第 {attempt} 次嘗試] 正在執行精準評測...")
    
    captured_output = io.StringIO()
    try:
        with redirect_stdout(captured_output):
            result = evaluate(
                data=DATASET_PATH, 
                evaluators={
                    "groundedness": groundedness_eval,
                    "relevance": relevance_eval,
                    "retrieval": retrieval_eval
                },
                # Column mapping:
                evaluator_config={
                    "groundedness": {
                        "column_mapping": {
                            "query": "${data.query}",
                            "context": "${data.context}",
                            "response": "${data.response}"
                        } 
                    }
                },
                # Optionally, provide your Foundry project information to track your evaluation results in your project portal.
                #azure_ai_project = azure_ai_project,
                # Optionally, provide an output path to dump a JSON file of metric summary, row-level data, and the metric and Foundry project URL.
                output_path=input_json_path,
                max_workers=1,
                raise_exceptions=True,
            )
    except Exception as e:
        # 即使 evaluate() 噴大錯被我們捕捉，Console 文字依然有留著，所以我們 continue 讓下面繼續解析
        print(f"⚠️ 評測引擎觸發異常，準備檢查 Console 日誌...")

    # 2. 撈出 Console 的所有純文字並解析錯誤行號
    console_logs = captured_output.getvalue()

    log_filename = f"eval_debug_round_{attempt}.log"
    with open(log_filename, "w", encoding="utf-8") as f_log:
        f_log.write(console_logs)
    
    log_filename = f"eval_metric_round_{attempt}.json"
    with open(input_json_path, "r", encoding="utf-8") as f_in:
        raw_data = json.load(f_in)
    with open(log_filename, "w", encoding="utf-8") as f_log:
        json.dump(raw_data, f_log, ensure_ascii=False, indent=4)
    

    this_round_failed_relative_indices = set()
    try:
        start_idx = console_logs.find("{")
        end_idx = console_logs.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            eval_dict = json.loads(console_logs[start_idx:end_idx])
            for metric in ["groundedness", "relevance", "retrieval"]:
                errors = eval_dict.get(metric, {}).get("per_line_errors", {})
                if errors:
                    #this_round_failed_relative_indices.update(errors.keys())
                    this_round_failed_relative_indices.update([int(k) for k in errors.keys()])
    except Exception as parse_error:
        print(f"🚨 解析 Console JSON 失敗: {parse_error}")

    with open(input_json_path, "r", encoding="utf-8") as f_this_round:
        this_round_json_data = json.load(f_this_round)
    this_round_rows = this_round_json_data.get("rows", [])


    if attempt == 1:
        final_total_rows = [{}] * total_lines # 第一次跑，先建立空總表
        global_metrics_data = dict(this_round_json_data)
        global_metrics_data["rows"] = [{}] * total_lines
    else:
        with open(output_jsonl_path, "r", encoding="utf-8") as f_in:
            final_total_rows = [json.loads(line) for line in f_in]
        if os.path.exists(GLOBAL_METRICS_JSON_PATH):
            with open(GLOBAL_METRICS_JSON_PATH, "r", encoding="utf-8") as f_global:
                global_metrics_data = json.load(f_global)
        else:
            global_metrics_data = {"rows": [{}] * total_lines}
        
    for relative_idx, global_idx in enumerate(current_round_global_indices):
        # 只有在這一輪這個相對位置「成功」時，才更新總表
        if relative_idx not in this_round_failed_relative_indices:
            # 假設這輪跑完的單列結果叫 row_result
            # final_total_rows[global_idx] = row_result
            #pass
            if relative_idx < len(this_round_rows):
                final_total_rows[global_idx] = this_round_rows[relative_idx]
                global_metrics_data["rows"][global_idx] = this_round_rows[relative_idx]

    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for row in final_total_rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(GLOBAL_METRICS_JSON_PATH, "w", encoding="utf-8") as f_global_out:
        json.dump(global_metrics_data, f_global_out, ensure_ascii=False, indent=4)
    
    # 3. 💥 核心邏輯：判斷是否有失敗行，進行備份與精準覆蓋
    if len(this_round_failed_relative_indices) > 0:
        # 轉換成真正大總管的行號，印出來給你看
        next_round_global_indices = [current_round_global_indices[i] for i in this_round_failed_relative_indices]
        print(f"❌ 本輪重跑後，依然有 {len(next_round_global_indices)} 筆頑固 case 失敗。真實行號：{next_round_global_indices}")
        
        if attempt >= attempt_limit:
            print(f"\n🛑 [達到重試上限] 已嘗試了 {attempt} 次，依然有 {len(next_round_global_indices)} 筆資料失敗。")
            print(f"🚨 頑固失敗的真實資料行號為: {next_round_global_indices}")
            print("💡 建議停止程式，手動檢查這些 JSONL 欄位內容是否有錯（如 response 為空或格式不對）。")
            break  # 強制跳出 while 迴圈，終止重試

        # 重新生成下一輪的局部 JSONL
        failed_data_buffer = []
        with open(BACKUP_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in next_round_global_indices:
                    failed_data_buffer.append(line.strip())
                    
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            for line_data in failed_data_buffer:
                f.write(line_data + "\n")
                
        # 遞補指引：下一輪的基礎對照表，更新為這一次失敗的全球行號
        current_round_global_indices = next_round_global_indices
        continue
    else:
        print("\n🎉🎉🎉【地獄難度全通關】！！！")
        print("✅ 歷經多輪慘烈重跑，所有資料都拿到了成功的數據，大總管總表已完美拼裝完成！")
        break


if os.path.exists(BACKUP_PATH):
    shutil.copy(BACKUP_PATH, DATASET_PATH)
    print("\n🔄 [自動還原] 已將原始完整的 11 筆資料還原回 ai_foundry_eval_dataset.jsonl")

groundedness_passed = 0
relevance_passed = 0
retrieval_passed = 0
with open(GLOBAL_METRICS_JSON_PATH, "r", encoding="utf-8") as f_out:
    global_data = json.load(f_in)
    global_rows = global_data.get("rows", [])
    for idx, row in enumerate(global_rows):
        #print(idx)
        g_pass = 1 if row.get("outputs.groundedness.groundedness_passed") is True else 0
        rel_pass = 1 if row.get("outputs.relevance.relevance_passed") is True else 0
        ret_pass = 1 if row.get("outputs.retrieval.retrieval_passed") is True else 0
        
        #print(f"num = {total_num}, g_pass = {g_pass}, rel_pass = {rel_pass}, ret_pass = {ret_pass}")

        groundedness_passed += g_pass
        relevance_passed += rel_pass
        retrieval_passed += ret_pass

        json_line = json.dumps(row, ensure_ascii=False)
        f_out.write(json_line + "\n")

evaluators_to_check = ["groundedness", "relevance", "retrieval"]
groundedness_errors = result.get("groundedness", {}).get("per_line_errors", {})
relevance_errors = result.get("relevance", {}).get("per_line_errors", {})
retrieval_errors = result.get("retrieval", {}).get("per_line_errors", {})
print(f"groundedness_errors: {groundedness_errors}")
print(f"relevance_errors: {relevance_errors}")
print(f"retrieval_errors: {retrieval_errors}")

# compute pass rate



        
#print(f"groundedness = {groundedness_passed*100/total_cases}%, relevance = {relevance_passed*100/total_cases}%, retrieval = {retrieval_passed*100/total_cases}%")
