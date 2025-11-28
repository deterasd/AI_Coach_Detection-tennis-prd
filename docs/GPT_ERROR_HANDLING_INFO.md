# GPT 錯誤處理說明

## 📋 修改摘要

已為 AI 網球教練系統新增 GPT API 錯誤容錯機制，確保在 GPT 配額不足或 API 錯誤時，系統能夠繼續處理所有球的分析。

## 🔧 修改的檔案

### 1. `trajectory_gpt_single_feedback.py`
**修改功能**: `generate_feedback_data_only()`

#### 原始行為:
- API 錯誤時會拋出異常
- 中斷整個處理流程

#### 新行為:
```python
# 第一次 API 呼叫（建議生成）
try:
    response = create_chat_completion(messages)
    knn_response = response.choices[0].message.content
except Exception as e:
    # 檢測配額錯誤 (429, quota exceeded, rate_limit)
    if "429" in error_msg or "quota" in error_msg.lower():
        return {
            "problem_frame": "0-0",
            "suggestion": f"KNN分析結果: {knn_feedback}\n(註: GPT配額不足，僅顯示KNN分析)",
            "error": True,
            "error_type": "quota_exceeded"
        }
    # 其他 API 錯誤
    else:
        return {
            "problem_frame": "0-0",
            "suggestion": f"KNN分析結果: {knn_feedback}\n(註: GPT暫時無法使用)",
            "error": True,
            "error_type": "api_error"
        }
```

#### 新增錯誤處理:
- ✅ 捕獲 API 配額錯誤 (HTTP 429)
- ✅ 捕獲其他 API 錯誤
- ✅ 返回 KNN 分析結果作為替代
- ✅ 標記錯誤類型 (`quota_exceeded`, `api_error`)
- ✅ 第二次 API 呼叫（frame range）也新增錯誤處理

### 2. `trajector_processing_unified.py`
**修改功能**: `process_single_video_set()` 中的 GPT 步驟

#### 新增的錯誤處理邏輯:
```python
# 步驟11：GPT反饋生成（帶錯誤容錯）
try:
    trajectory_gpt_suggestion = generate_feedback_data_only(...)
    
    # 檢查是否有錯誤標記
    if trajectory_gpt_suggestion.get('error', False):
        error_type = trajectory_gpt_suggestion.get('error_type', 'unknown')
        if error_type == 'quota_exceeded':
            print("⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代")
        else:
            print(f"⚠️ GPT API 發生錯誤，已使用 KNN 分析結果作為替代")
    
    # 即使有錯誤也保存替代結果
    gpt_feedback_path = save_gpt_feedback_with_output_folder(...)
    
except Exception as e:
    print(f"⚠️ GPT反饋生成失敗: {e}")
    print("⚠️ 跳過 GPT 步驟，繼續處理...")
    
    # 創建簡單的反饋結果
    trajectory_gpt_suggestion = {
        "problem_frame": "N/A",
        "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
        "error": True
    }
```

### 3. `trajector_processing_unified.py` - 多球處理
**修改功能**: `process_multiple_balls()` 中的球循環

#### 修改重點:
```python
for ball_number in ball_pairs:
    try:
        success = process_single_video_set(...)
        
        if success:
            print(f"✅ 第 {ball_number} 顆球處理完成")
        else:
            print(f"⚠️ 第 {ball_number} 顆球處理有部分問題，但已完成可執行的步驟")
            # 不中斷，繼續處理下一顆球
            
    except Exception as e:
        print(f"❌ 第 {ball_number} 顆球處理發生錯誤: {e}")
        print(f"⚠️ 跳過第 {ball_number} 顆球，繼續處理下一顆...")
        # 不中斷，繼續處理下一顆球
```

## 🎯 新行為說明

### GPT 配額不足時:
1. ✅ 系統偵測到 HTTP 429 或 "quota exceeded" 錯誤
2. ✅ 自動使用 KNN 分析結果作為替代
3. ✅ 在回饋檔案中註明: "GPT配額不足，僅顯示KNN分析"
4. ✅ **繼續處理下一顆球**，不中斷流程

### 其他 API 錯誤時:
1. ✅ 捕獲所有 OpenAI API 錯誤
2. ✅ 使用 KNN 分析結果作為替代
3. ✅ 在回饋檔案中註明錯誤訊息
4. ✅ **繼續處理下一顆球**，不中斷流程

### 生成的檔案範例:

#### 正常情況 (`_gpt_feedback.json`):
```json
{
  "problem_frame": "15-30",
  "suggestion": "您的擊球動作整體不錯，但在揮拍時手腕角度稍微偏高..."
}
```

#### GPT 配額不足時 (`_gpt_feedback.json`):
```json
{
  "problem_frame": "0-0",
  "suggestion": "KNN分析結果: 頭:沒問題!、肩膀:沒問題!、手腕:角度偏高...\n(註: GPT配額不足，僅顯示KNN分析)",
  "error": true,
  "error_type": "quota_exceeded"
}
```

## 🔍 除錯資訊

### 控制台輸出範例:

#### 第 1 顆球 - GPT 正常:
```
處理第 1 顆球...
步驟11：生成GPT反饋...
✅ GPT反饋生成完成，耗時：2.3456 秒
✅ 第 1 顆球處理完成
```

#### 第 2 顆球 - GPT 配額不足:
```
處理第 2 顆球...
步驟11：生成GPT反饋...
⚠️ GPT API 回應錯誤: 429 - quota exceeded
⚠️ GPT API 配額不足，使用 KNN 分析結果作為替代
⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代
✅ GPT反饋生成完成，耗時：0.1234 秒
✅ 第 2 顆球處理完成
```

#### 第 3 顆球 - 繼續處理:
```
處理第 3 顆球...
步驟11：生成GPT反饋...
⚠️ GPT API 回應錯誤: 429 - quota exceeded
⚠️ GPT API 配額不足，使用 KNN 分析結果作為替代
⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代
✅ GPT反饋生成完成，耗時：0.0987 秒
✅ 第 3 顆球處理完成

🎾 所有球對分析完成！共處理 3 個球對
```

## 📊 處理結果

### 所有球都會被處理:
- 第 1 顆球: ✅ 完整處理（包含 GPT 建議）
- 第 2 顆球: ⚠️ 使用 KNN 結果（GPT 配額不足）
- 第 3 顆球: ⚠️ 使用 KNN 結果（GPT 配額不足）
- **所有球都有分析結果，沒有遺漏**

### 生成的資料夾結構:
```
trajectory/(姓名)__trajectory/
├── trajectory_1/
│   ├── (姓名)__1_segment_gpt_feedback.json  ← GPT 建議
│   └── ... (其他檔案)
├── trajectory_2/
│   ├── (姓名)__2_segment_gpt_feedback.json  ← KNN 分析結果（GPT 替代）
│   └── ... (其他檔案)
└── trajectory_3/
    ├── (姓名)__3_segment_gpt_feedback.json  ← KNN 分析結果（GPT 替代）
    └── ... (其他檔案)
```

## ✨ 優勢

1. **不中斷處理流程**: 即使 GPT 失敗，仍會處理所有球
2. **提供替代方案**: 使用 KNN 分析結果作為替代
3. **清楚的錯誤訊息**: 明確告知使用者為何使用替代方案
4. **完整的資料保存**: 所有分析結果都會被保存
5. **易於除錯**: 詳細的控制台輸出和錯誤標記

## 🚀 測試建議

使用 `trajector_processing_simple_test.py` 進行測試：

```powershell
python trajector_processing_simple_test.py
```

即使 GPT 配額用盡，系統也會：
- ✅ 處理所有偵測到的球
- ✅ 生成完整的 2D/3D 軌跡
- ✅ 提供 KNN 分析結果
- ✅ 創建所有必要的資料夾和檔案
- ✅ 在報告中註明 GPT 功能暫時無法使用

## 📝 注意事項

1. **KNN 分析仍會執行**: 即使 GPT 失敗，KNN 分析仍會正常運作
2. **不影響其他功能**: 2D/3D 軌跡、影片處理等功能完全不受影響
3. **錯誤標記**: 可以透過檢查 `error` 欄位判斷是否使用替代方案
4. **配額恢復**: 當 GPT 配額恢復後，可以重新執行以獲得完整的 GPT 建議
