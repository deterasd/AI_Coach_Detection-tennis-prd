import os
import pandas as pd
import json
import time
import re
import single_feedback.prompt as prompt
import single_feedback.model_config as model_config
from openai import OpenAI
from ai_config import ai_config

# --- 設定 API 參數與載入 Prompt 與模型設定 ---
client = ai_config.get_client()
MODEL = ai_config.get_model_name()  # 從 ai_config 動態獲取正確的模型名稱
TEMPERATURE = model_config.TEMPERATURE
MAX_TOKENS = model_config.MAX_TOKENS
FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
TOP_P = model_config.TOP_P

INSTRUCTIONS = prompt.INSTRUCTIONS
DATADESCIRBE = prompt.DATADESCIRBE

# 簡化系統提示詞以提升速度和穩定性
system_content = """你是專業的網球教練。
請根據 KNN 分析結果，用 2 句話描述需要改進的動作部位與方向。
回覆必須使用繁體中文，語氣友善。"""

def create_chat_completion(messages):
    """
    以給定的 messages 呼叫 ChatCompletion（支援 LM Studio 或 OpenAI）
    回傳產生的 completion 結果
    """
    if client is None:
        print("❌ AI 客戶端未初始化")
        return None

    request_kwargs = dict(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )

    # 優化參數設定以提升速度
    if ai_config.is_lm_studio():
        # LM Studio 本地模型設定 - 使用優化參數
        print(f"🤖 使用 LM Studio 本地模型: {MODEL}")
        request_kwargs["max_tokens"] = 100  # 降低至 100 以加快生成
        request_kwargs["temperature"] = 0.5  # 降低至 0.5 提升速度
        request_kwargs["top_p"] = 0.9  # 降低至 0.9
    else:
        # OpenAI API 設定
        print(f"🌐 使用 OpenAI API 模型: {MODEL}")
        model_lower = MODEL.lower()
        if any(prefix in model_lower for prefix in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3")):
            request_kwargs["max_completion_tokens"] = MAX_TOKENS
            if any(prefix in model_lower for prefix in ("gpt-5", "o1", "o3")):
                request_kwargs["temperature"] = 1
                request_kwargs.pop("frequency_penalty", None)
                request_kwargs.pop("presence_penalty", None)
                request_kwargs.pop("top_p", None)
        else:
            request_kwargs["max_tokens"] = MAX_TOKENS

    try:
        completion = client.chat.completions.create(**request_kwargs)
        return completion
    except Exception as e:
        print(f"❌ API 呼叫失敗: {e}")
        if ai_config.is_lm_studio():
            print("💡 請確認 LM Studio 已啟動且載入模型")
            print("💡 或者執行 ai_config.switch_to_openai() 切換到 OpenAI API")
        else:
            print("💡 請檢查網路連接和 API 金鑰")
        return None

def generate_feedback(json_filepath, txt_filepath):
    """
    讀取 JSON (運動軌跡) 與 KNN 結果(txt)，並綜合兩者資訊產出 GPT 回饋
    最後將結果輸出為 _gpt_feedback.json 檔
    """
    # 讀取運動軌跡資料與 KNN 回饋
    my_motion = pd.read_json(json_filepath)
    knn_feedback = pd.read_csv(txt_filepath, header=None).iloc[0, 0]

    # 初始化 messages 列表 (LM Studio 不支援 system 角色，將內容合併到第一個 user 訊息)
    messages = []

    # 如果 knn_feedback 為特定正向回饋訊息
    if knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、是否擊球:是、其他:無":
        knn_response = "沒有觀察到顯著問題，請繼續保持！"
        frame_response = "0-0"

        # 將 frame 與建議回饋一起附加到 messages 中
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    else:
        # 第一次讓 GPT 根據 KNN Feedback 產生中文敘述
        messages.append({
            "role": "user",
            "content": system_content + f"""

                observe analysis results: {knn_feedback}, 
                Rephrase the analysis results of each body part in 1 sentence
            """
        })
        knn_completion = create_chat_completion(messages)
        knn_response = knn_completion.choices[0].message.content

        # 根據 KNN 回饋推測問題影格範圍（不需要完整的軌跡數據）
        total_frames = len(my_motion)
        messages.append({
            "role": "user",
            "content": f"""
                The feedback describes issues in a tennis swing motion with {total_frames} total frames.
                Based on the feedback: "{knn_response}", 
                speculate in which frame section the issue most likely occurs. 
                Please provide a broader frame range covering more frames (e.g., a range of at least 15 frames), 
                and You MUST respond with a numeric range only, in the format "number-number" (e.g., "13-24"), 
                containing only digits and a hyphen, with no additional text or formatting.
            """
        })
        frame_completion = create_chat_completion(messages)
        frame_response = frame_completion.choices[0].message.content

        # 將數字範圍與 knn_response 加入到 messages (可以用於後續檢視或除錯)
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    # 處理換行符號
    frame_response = frame_response.replace("\n", "")
    knn_response = knn_response.replace("\n", "")

    # 構造 JSON 格式回傳結果
    ai_feedback = {
        "problem_frame": frame_response,
        "suggestion": knn_response,
    }

    print(ai_feedback)

    # 輸出檔案路徑 (以原檔案名稱 + "_gpt_feedback.json")
    output_filepath = json_filepath.replace('(3D_trajectory_smoothed)_only_swing.json', '_gpt_feedback.json')
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(ai_feedback, f, ensure_ascii=False, indent=2)

    return output_filepath

def generate_feedback_data_only(json_filepath, txt_filepath):
    """
    讀取 JSON (運動軌跡) 與 KNN 結果(txt)，並綜合兩者資訊產出 GPT 回饋
    返回數據而不保存檔案
    如果 API 配額不足或其他錯誤，返回包含錯誤訊息的回應
    """
    try:
        # 讀取運動軌跡資料 (使用 json.load 代替 pd.read_json)
        with open(json_filepath, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        # 轉換為 DataFrame (如果需要的話)
        my_motion = pd.DataFrame(trajectory_data)
        
        # 讀取 KNN 回饋
        knn_feedback = pd.read_csv(txt_filepath, header=None).iloc[0, 0]
    except Exception as e:
        print(f"⚠️ 讀取資料失敗: {e}")
        return {
            "problem_frame": "N/A",
            "suggestion": f"資料讀取失敗: {str(e)}",
            "error": True
        }

    total_frames = len(my_motion)
    frame_response = "0-0"

    # 特殊情況：完全無問題的 KNN 回饋直接回傳固定訊息
    if knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、是否擊球:是、其他:無":
        knn_response = "沒有觀察到顯著問題，請繼續保持！"
    else:
        knn_messages = [
            {
                "role": "user",
                "content": f"{system_content}\n\nKNN 分析結果：\n{knn_feedback}"
            }
        ]

        try:
            knn_completion = create_chat_completion(knn_messages)
            knn_response = knn_completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ GPT API 回應錯誤: {error_msg}")

            if "429" in error_msg or "quota" in error_msg.lower() or "rate_limit" in error_msg.lower():
                print("⚠️ GPT API 配額不足，使用 KNN 分析結果作為替代")
                return {
                    "problem_frame": "0-0",
                    "suggestion": f"KNN分析結果: {knn_feedback}\n(註: GPT配額不足，僅顯示KNN分析)",
                    "error": True,
                    "error_type": "quota_exceeded"
                }
            else:
                print("⚠️ GPT API 發生其他錯誤，使用 KNN 分析結果作為替代")
                return {
                    "problem_frame": "0-0",
                    "suggestion": f"KNN分析結果: {knn_feedback}\n(註: GPT暫時無法使用 - {error_msg})",
                    "error": True,
                    "error_type": "api_error"
                }

        # 取得動作問題所在的影格範圍（簡化：固定回傳預設值）
        # 移除耗時的影格推測步驟以提升整體速度
        frame_response = "0-0"

    # 正規化輸出內容
    knn_response = (knn_response or "").replace("\n", " ").strip()

    frame_response = (frame_response or "0-0").strip()
    match = re.search(r"(\d+)\s*-\s*(\d+)", frame_response)
    if match:
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        if total_frames > 0:
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))
        frame_response = f"{start_frame}-{end_frame}"
    else:
        frame_response = "0-0"

    ai_feedback = {
        "problem_frame": frame_response,
        "suggestion": knn_response or f"KNN分析結果: {knn_feedback}",
    }

    return ai_feedback


if __name__ == "__main__":
    json_path = "嘉洋__3(3D_trajectory_smoothed).json"
    txt_path = "嘉洋__3_knn_feedback.txt"

    # 開始計時
    start_time = time.time()

    # 產生並輸出回饋
    output_filepath = generate_feedback(json_path, txt_path)

    # 結束計時
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("AI Feedback:")
    print(f"Processing time: {elapsed_time:.2f} seconds")