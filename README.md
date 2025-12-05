# AI Coach Validation Dashboard (3D Tennis Analysis)

這是一個用於驗證 3D 網球動作分析系統的視覺化控制台。它提供了一個網頁介面，讓開發者與研究人員能夠驗證 3D 重建數據的準確性、骨架一致性以及其他物理合理性指標。

## 功能特色

- **視覺化儀表板**: 透過現代化的 Dark Mode / Glassmorphism 介面進行操作。
- **多步驟驗證流程**:
  - **Step 1: 重投影誤差驗證 (Reprojection Error)**: 檢查 3D 重建結果投影回 2D 畫面時的誤差，確保相機參數與重建演算法的準確度。
  - **Step 2: 骨架一致性驗證 (Bone Consistency)**: 分析骨骼長度在時間序列上的變異，確保生物力學的合理性。
- **一鍵自動化**: 支援單步執行或一鍵執行所有測試。
- **詳細報告**: 每個驗證步驟皆會生成詳細的 HTML 報告，包含圖表與統計數據。

## 專案架構

```
AI_Coach_Detection-tennis/
├── modules/                  # 核心分析模組
│   ├── step1_reprojection_error.py
│   ├── step2_bone_consistency.py
│   ├── utils.py              # 共用工具函式
│   └── ...
├── templates/                # 網頁前端模板
│   ├── dashboard.html        # 主控制台
│   ├── step1_reprojection_error.html
│   └── step2_bone_consistency.html
├── data/                     # 測試資料存放區 (JSON)
├── config.py                 # 設定檔
├── main.py                   # Flask 伺服器入口
├── requirements.txt          # Python 依賴套件
└── validation_config.json    # 驗證參數設定
```

## 安裝與執行

### 1. 環境需求

請確保已安裝 Python 3.8 或以上版本。

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 啟動伺服器

```bash
python main.py
```

伺服器啟動後，請在瀏覽器中開啟：
`http://localhost:5000`

## 使用說明

1. **資料來源選擇**:
   - 在儀表板左側選擇包含測試數據的資料夾。
   - 系統會自動配對 3D JSON、2D Side JSON 與 2D 45° JSON 檔案。

2. **相機參數設定**:
   - 確認或輸入 P1 (Side Camera) 與 P2 (45° Camera) 的投影矩陣。

3. **執行驗證**:
   - 點擊「執行分析」按鈕來執行個別步驟。
   - 或點擊「🚀 一鍵執行全部」來自動跑完所有流程。
   - 完成後點擊「查看報告」以檢視詳細分析結果。

## 開發者資訊

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla)
- **Analysis**: NumPy, SciPy
