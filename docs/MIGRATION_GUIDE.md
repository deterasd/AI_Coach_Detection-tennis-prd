# GPT 轉 LM Studio 完整遷移指南

## 📋 總結概述

這份指南將幫助您將原始 GPT 專案遷移到支援 LM Studio 的智能切換系統。

## 🎯 遷移目標

- ✅ 支援 LM Studio 本地模型
- ✅ 支援遠程 LM Studio (透過 ngrok)
- ✅ 保持 GPT API 作為備援
- ✅ 透過 config.json 動態切換
- ✅ 速度優化配置
- ✅ 自動錯誤處理和回退

## 📁 需要新增的檔案 (共5個)

### 1️⃣ ai_config.py
**作用**：AI 模型自動切換和管理核心
**狀態**：✅ 已完成
```python
# 主要功能：
- 自動偵測 LM Studio 連接
- 讀取 config.json 配置
- 智能切換 LM Studio ↔ OpenAI
- 動態模型名稱管理
- 錯誤處理和自動回退
```

### 2️⃣ remote_lm_studio_config.py
**作用**：遠程 LM Studio 連接配置
**狀態**：✅ 已完成
```python
REMOTE_LM_STUDIO_URL = "https://your-ngrok-url.ngrok-free.dev/v1"
REMOTE_LM_STUDIO_API_KEY = "lm-studio"
```

### 3️⃣ switch_model.py
**作用**：快速切換 AI 模型的命令行工具
**狀態**：✅ 已完成
```bash
# 使用方法：
python switch_model.py gpt-4o        # 切換到 GPT-4o
python switch_model.py gpt-4o-mini   # 切換到 GPT-4o-mini
python switch_model.py lm-studio     # 切換到 LM Studio
python switch_model.py auto          # 自動偵測模式
python switch_model.py status        # 查看目前狀態
```

### 4️⃣ speed_optimizer.py
**作用**：AI 回應速度優化工具
**狀態**：✅ 已完成
```bash
# 使用方法：
python speed_optimizer.py ultra_fast  # 極速模式 (0.5-1秒)
python speed_optimizer.py fast        # 快速模式 (1-1.5秒)
python speed_optimizer.py balanced    # 平衡模式 (1.5-2秒)
python speed_optimizer.py quality     # 品質模式 (2-3秒)
```

### 5️⃣ test_ai_switch.py
**作用**：測試 AI 切換功能和連接狀態
**狀態**：✅ 已完成
```bash
# 使用方法：
python test_ai_switch.py  # 測試當前 AI 配置
```

## 🔄 需要修改的檔案 (共3個)

### 1️⃣ config.json
**原始狀態**：可能不存在，或只有基本設定
**需要新增**：AI 模型配置區塊
```json
{
  "ball_direction": "right",
  "confidence_threshold": 0.5,
  "last_updated": "2025-11-20",
  "ai_model": {
    "provider": "auto",                    # 新增：模型提供者
    "model_name": "auto",                  # 新增：模型名稱
    "temperature": 0.3,                    # 新增：溫度參數
    "max_tokens": 50,                      # 新增：最大 token 數
    "top_p": 0.8,                         # 新增：top_p 參數
    "frequency_penalty": 0.0,              # 新增：頻率懲罰
    "presence_penalty": 0.0,               # 新增：存在懲罰
    "fallback_enabled": true,              # 新增：啟用自動回退
    "fallback_order": ["lm-studio", "gpt-4o-mini", "gpt-4o"]  # 新增：回退順序
  }
}
```

### 2️⃣ trajectory_gpt_single_feedback.py
**需要修改的部分**：
```python
# 修改前：
from openai import OpenAI
from open_ai_key import api_key
client = OpenAI(api_key=api_key)
MODEL = model_config.MODEL

# 修改後：
from ai_config import ai_config
client = ai_config.get_client()
MODEL = ai_config.get_model_name()  # 動態獲取正確的模型名稱
```

### 3️⃣ single_feedback/model_config.py (選用)
**可能需要確認**：模型名稱設定
```python
MODEL = "google/gemma-3n-e4b"  # LM Studio 模型名稱
# 或
MODEL = "gpt-4o-mini"          # OpenAI 模型名稱
```

## 🚀 完整遷移步驟

### 步驟 1：準備檔案
```powershell
# 將以下檔案複製到原始專案根目錄：
- ai_config.py
- remote_lm_studio_config.py
- switch_model.py
- speed_optimizer.py
- test_ai_switch.py
```

### 步驟 2：更新配置
```powershell
# 1. 編輯或創建 config.json，加入 ai_model 區塊
# 2. 更新 remote_lm_studio_config.py 中的 ngrok URL
# 3. 確認 single_feedback/model_config.py 中的模型名稱
```

### 步驟 3：修改程式碼
```python
# 在 trajectory_gpt_single_feedback.py 中：
# 替換 OpenAI 初始化代碼為 ai_config 導入
```

### 步驟 4：設置 LM Studio
```bash
# 1. 安裝並啟動 LM Studio
# 2. 下載並載入模型 (例如：google/gemma-3n-e4b)
# 3. 啟動 Local Server (預設 port 1234)
# 4. 設置 ngrok: ngrok http 1234
# 5. 更新 remote_lm_studio_config.py 中的 URL
```

### 步驟 5：測試和驗證
```powershell
# 測試連接
python test_ai_switch.py

# 測試模型切換
python switch_model.py status
python switch_model.py lm-studio
python switch_model.py gpt-4o-mini

# 測試速度優化
python speed_optimizer.py fast
```

## ⚡ 速度優化配置

根據我們的測試，以下是最佳速度配置：

| 參數 | 極速模式 | 快速模式 | 平衡模式 | 品質模式 |
|------|----------|----------|----------|----------|
| temperature | 0.1 | 0.3 | 0.5 | 0.7 |
| max_tokens | 30 | 50 | 80 | 120 |
| top_p | 0.7 | 0.8 | 0.9 | 0.95 |
| 預期速度 | 0.5-1秒 | 1-1.5秒 | 1.5-2秒 | 2-3秒 |

## 🔧 故障排除

### 常見問題和解決方案：

1. **LM Studio 連接失敗**
   - 檢查 LM Studio 是否啟動
   - 確認模型已載入
   - 驗證 ngrok URL 正確性

2. **模型名稱錯誤**
   - 使用 `python switch_model.py status` 檢查
   - 確認 LM Studio 中的實際模型名稱

3. **速度太慢**
   - 使用 `python speed_optimizer.py fast` 或 `ultra_fast`
   - 確保使用 GPU 加速
   - 選擇較小的模型

4. **自動回退不工作**
   - 確認 config.json 中 fallback_enabled = true
   - 檢查 OpenAI API 金鑰是否有效

## ✅ 遷移檢查清單

**準備階段：**
- [ ] 備份原始專案
- [ ] 準備 LM Studio 環境
- [ ] 安裝 ngrok (如需遠程連接)

**檔案新增：**
- [ ] ai_config.py
- [ ] remote_lm_studio_config.py  
- [ ] switch_model.py
- [ ] speed_optimizer.py
- [ ] test_ai_switch.py

**檔案修改：**
- [ ] config.json (加入 ai_model 配置)
- [ ] trajectory_gpt_single_feedback.py (改用 ai_config)
- [ ] single_feedback/model_config.py (確認模型名稱)

**測試驗證：**
- [ ] LM Studio 本地連接測試
- [ ] LM Studio 遠程連接測試 (如適用)
- [ ] OpenAI API 備援測試
- [ ] 模型切換功能測試
- [ ] 速度優化測試
- [ ] 完整流程測試

**完成後效果：**
- ✅ 可透過 config.json 或命令行快速切換 AI 模型
- ✅ 自動偵測和回退機制
- ✅ 速度優化 (從 2-3秒 → 0.5-1.5秒)
- ✅ 支援本地和遠程 LM Studio
- ✅ 保持 GPT 作為備援選項

## 🎯 使用建議

**日常使用：**
```powershell
# 快速模式 (推薦)
python speed_optimizer.py fast

# LM Studio 本地模式 (最快，免費)
python switch_model.py lm-studio

# 需要高品質時使用 GPT
python switch_model.py gpt-4o-mini
```

**批量處理：**
```powershell
# 極速模式
python speed_optimizer.py ultra_fast
```

**重要分析：**
```powershell
# 品質模式
python speed_optimizer.py quality
python switch_model.py gpt-4o
```

遷移完成後，您將擁有一個智能、快速、可靠的 AI 模型切換系統！🎉