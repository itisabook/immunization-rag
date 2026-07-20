import pandas as pd
import json

def convert_csv_to_evaluation_jsonl(golden_csv_path, agent_csv_path, output_jsonl_path):
    """
    讀取 Golden 與 Agent 的 CSV 檔案，根據 Case_ID 進行對齊，
    並導出為 Azure AI Foundry 評估所需的 JSONL 格式。
    """
    # 1. 讀取兩個 CSV 檔案
    print("正在讀取 CSV 檔案...")
    if golden_csv_path:
        golden_df = pd.read_csv(golden_csv_path)
        golden_clean = golden_df[['Case_ID', 'ground_truth', 'context', 'query']].copy()
    
    if agent_csv_path:
        agent_df = pd.read_csv(agent_csv_path)    
        agent_clean = agent_df[['Case_ID', 'agent_response', 'agent_context', 'query']].copy()
    
    if golden_csv_path and agent_csv_path:
        print("正在根據 Case_ID 進行資料對齊...")
        merged_df = pd.merge(golden_clean, agent_clean, on='Case_ID', how='inner')
    elif golden_csv_path:
        merged_df = golden_clean
    elif agent_csv_path:
        merged_df = agent_clean
    else:
        print("Error: golden_csv_path and agent_csv_path are NULL!")
        return
    
    # 4. 轉換為 Azure AI Foundry Evaluation 標準的 JSONL 格式
    print(f"正在轉換並寫入 JSONL 格式...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in merged_df.iterrows():
            # 對應 AI Foundry 的內建評估指標（如 Groundedness, Relevance, Fluency 等）
            # 它們通常預設讀取: query, response, context, ground_truth
            eval_record = {
                "Case_ID": str(row['Case_ID']),
                "query": str(row['query']),
            }
            if 'agent_response' in row:
                eval_record["response"] = str(row['agent_response'])
            if 'agent_context' in row:
                eval_record["context"] = str(row['agent_context'])
            if 'ground_truth' in row:
                eval_record["ground_truth"] = str(row['ground_truth'])
            if 'context' in row:
                eval_record["golden_context"] = str(row['context'])
            # 寫入單行 JSON 
            f.write(json.dumps(eval_record, ensure_ascii=False) + '\n')
            
    print(f"\n==== 轉換成功 ====")
    print(f"輸出路徑: {output_jsonl_path}")
    print(f"成功對齊並轉換的總筆數: {len(merged_df)} 筆")

if __name__ == "__main__":
    # 請將下方的檔名替換為您本地實際的檔案路徑
    GOLDEN_CSV = []#"golden_test_benchmark_100_20260530_003852_20260601_171126.csv" # first version
    #"golden_test_benchmark_100_20260530_003852_20260602_130546.csv" # add sleep(2) between each LLM query
    #"golden_test_benchmark_100_20260530_003852_20260603_000129.csv" # sleep(2), modify prompts for librarian
    
    AGENT_CSV = "results_test_benchmark_25_20260522_025026_new_20260608_193504.csv"
    #"results_test_benchmark_100_20260530_003852_20260606_141506.csv" ## 3 steps, all gpt-5-mini
    #"results_test_benchmark_100_20260530_003852_20260605_093559.csv" # both gpt-4o, temperature = 0, 0
    #"results_test_benchmark_100_20260530_003852_20260605_081541.csv" # librarian=gpt-5, ai_audit=gpt-4o (temperature=0)
    #"results_test_benchmark_100_20260530_003852_20260604_234421.csv" # # librarian=gpt-4o (temperature=0), ai_audit=gpt-5
    #"results_test_benchmark_100_20260530_003852_20260604_211903.csv" # both gpt-5
    #"results_test_benchmark_100_20260530_003852_20260604_002155.csv" # both gpt-4o, change [KEY_EVIDENCE] [CLINICAL_ADVICE] prompts, temperature = 0, 0
    #"results_test_benchmark_100_20260530_003852_20260603_231408.csv" # librarian=gpt-5, ai_audit=gpt-4o (temperature=0)
    #"results_test_benchmark_100_20260530_003852_20260603_212902.csv" # librarian=gpt-4o (temperature=0), ai_audit=gpt-5
    #"results_test_benchmark_100_20260530_003852_20260601_015039.csv" # temperature = 0.1, 0.1
    #"results_test_benchmark_100_20260530_003852_20260601_224149.csv" # first version, temperature = 0, 0
    
    OUTPUT_JSONL = "ai_foundry_eval_dataset.jsonl"
    
    print("golden: ", GOLDEN_CSV, "\nagent: ", AGENT_CSV)
    convert_csv_to_evaluation_jsonl(GOLDEN_CSV, AGENT_CSV, OUTPUT_JSONL)