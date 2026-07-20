import os
import re

def merge_and_clean_manual(root_dir, output_file):
    combined_content = []
    # 正則表達式：匹配 Markdown 中的 Base64 圖片編碼
    # 它會尋找 ![](data:image...base64...) 這種格式
    base64_pattern = r'\(data:image\/.*?;base64,.*?\)'
    
    file_count = 0
    removed_images_count = 0

    print("🚀 開始掃描並合併檔案...")

    # 1. 遍歷目錄
    for root, dirs, files in os.walk(root_dir):
        # 排序確保邏輯順序（例如 chunk_116 在 chunk_136 之前）
        files.sort()
        for filename in sorted(files):
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                print(file_path)
                file_count += 1
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # 2. 清理 Base64 圖片字串
                    # 找出有多少張圖片被清理（可選，僅供 Debug）
                    images_in_file = len(re.findall(base64_pattern, content))
                    removed_images_count += images_in_file
                    
                    # 執行替換，將亂碼換成簡短的提示文字
                    clean_content = re.sub(base64_pattern, '(image_data_removed)', content)
                    
                    # 3. 存入緩存
                    combined_content.append(f"\n\n")
                    combined_content.append(clean_content)
                    combined_content.append("\n\n---\n\n")

    # 4. 寫入最終檔案
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(combined_content)

    print("-" * 30)
    print(f"✅ 合併完成！")
    print(f"📂 總共處理檔案數: {file_count}")
    print(f"✂️  總共移除圖片亂碼: {removed_images_count} 處")
    print(f"💾 最終檔案: {output_file}")

# 執行 (請將路徑替換為你 marker 輸出的根目錄)
merge_and_clean_manual(r"c:\Users\Owner\Documents\compsci714\hackathon\immunization_guidelines\handbook_full_md", "NZ_Immunisation_Handbook_2026_by_marker_original.md")