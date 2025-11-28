# LM Studio 本地模型設定指南

## 📋 概述

本專案現在支援使用 LM Studio 載入本地大語言模型，完全離線運作，不需要網路連接和 OpenAI API 金鑰。

## 🚀 快速開始

### 1. 安裝 LM Studio

1. 下載並安裝 [LM Studio](https://lmstudio.ai/)
2. 啟動 LM Studio 應用程式

### 2. 下載和載入模型

1. 在 LM Studio 中搜尋並下載適合的模型：
   - **推薦模型**：`llama-2-7b-chat`, `mistral-7b-instruct`, `codellama-7b-instruct`
   - **輕量模型**：`phi-2`, `orca-mini-7b`

2. 載入模型到 LM Studio

### 3. 啟動服務器

#### 本地使用：
1. 在 LM Studio 中點擊 **"Start Server"** 按鈕
2. 確認服務器運行在 `http://localhost:1234`

#### 遠程使用（通過 ngrok）：
1. 在 LM Studio 電腦上安裝 [ngrok](https://ngrok.com/)
2. 啟動 ngrok：`ngrok http 1234`
3. 複製生成的 URL（例如：`https://abc123.ngrok.io`）
4. 在您的程式電腦上編輯 `remote_lm_studio_config.py`：
   ```python
   REMOTE_LM_STUDIO_URL = "https://abc123.ngrok.io/v1"  # 您的 ngrok URL
   REMOTE_LM_STUDIO_API_KEY = "lm-studio"  # 如果需要驗證
   ```

### 4. 配置專案

#### 修改模型設定
編輯 `single_feedback/model_config.py`：

```python
# 將 MODEL 設為您在 LM Studio 中載入的模型名稱
MODEL = "llama-2-7b-chat"  # 或者其他您載入的模型名稱
```

#### 調整參數（可選）
```python
TEMPERATURE = 0.7      # 創造性程度 (0.1-1.0)
MAX_TOKENS = 1000      # 最大回應長度
TOP_P = 0.9           # 核取樣參數
```

## 🧪 測試配置

運行測試腳本驗證設定：

```bash
python test_lm_studio.py
```

測試腳本會：
- ✅ 檢查 LM Studio 連接
- ✅ 測試聊天功能
- ✅ 驗證 OpenAI API fallback

## 🔄 自動切換機制

系統會自動選擇可用的 AI 服務：

1. **優先使用 LM Studio**：如果本地服務器可用
2. **自動 fallback 到 OpenAI**：如果 LM Studio 不可用
3. **手動切換**：可程式化切換

```python
from ai_config import ai_config

# 檢查當前狀態
if ai_config.is_lm_studio():
    print("使用 LM Studio")
else:
    print("使用 OpenAI API")

# 手動切換
ai_config.switch_to_lm_studio()   # 切換到 LM Studio
ai_config.switch_to_openai()      # 切換到 OpenAI
```

## 📁 修改的檔案

### 核心檔案
- **`ai_config.py`** - 新增的 AI 配置管理器
- **`trajectory_gpt_single_feedback.py`** - 修改為支援 LM Studio
- **`single_feedback/model_config.py`** - 更新模型設定
- **`test_lm_studio.py`** - 新增的測試腳本

### 設定方式
1. **LM Studio 優先**：系統預設嘗試連接本地 LM Studio
2. **OpenAI 備用**：如果本地連接失敗，自動使用 OpenAI API
3. **無縫切換**：程式運行中可動態切換 AI 服務

## ⚙️ 進階配置

### 環境變數設定
可以通過環境變數設定 OpenAI API 金鑰：

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

### 自訂 LM Studio 端點
如果 LM Studio 運行在不同端口，修改 `ai_config.py`：

```python
self.lm_studio_url = "http://localhost:8080/v1"  # 自訂端口
```

## 🔧 故障排除

### 常見問題

**Q: 無法連接 LM Studio**
- ✅ 確認 LM Studio 已啟動
- ✅ 確認服務器運行在正確端口
- ✅ 檢查防火牆設定

**Q: 模型載入失敗**
- ✅ 確認模型已下載並載入到 LM Studio
- ✅ 確認模型名稱正確
- ✅ 嘗試重新啟動 LM Studio

**Q: 回應品質不佳**
- ✅ 調整 `TEMPERATURE` 參數 (0.1-1.0)
- ✅ 嘗試不同的模型
- ✅ 增加 `MAX_TOKENS` 值

### 效能優化

- **輕量模型**：使用 7B 參數的模型以獲得更好效能
- **GPU 加速**：確保 LM Studio 能使用 GPU
- **記憶體管理**：監控系統記憶體使用量

## 📊 比較表

| 特性 | LM Studio | OpenAI API |
|------|-----------|------------|
| **成本** | 免費（硬體成本） | 付費 |
| **網路** | 離線使用 | 需要網路 |
| **隱私** | 本地處理 | 雲端處理 |
| **速度** | 依硬體而定 | 通常較快 |
| **可用性** | 24/7（本地） | 依服務狀態 |

## 🎯 使用建議

- **開發測試**：使用 LM Studio 節省 API 費用
- **生產環境**：視需求選擇 LM Studio 或 OpenAI
- **離線場景**：LM Studio 最適合沒有網路的環境
- **客製化**：LM Studio 可載入微調過的模型

---

如有問題，請參考 `test_lm_studio.py` 的測試結果進行故障排除。