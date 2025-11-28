# 🚨 測試過程問題分析與解決方案

## 📊 問題摘要

在測試過程中遇到了兩個主要問題：

### ❌ 問題1: FFmpeg 未安裝
```
FFmpeg 分割失敗: [WinError 2] 系統找不到指定的檔案。
```

### ❌ 問題2: 影片處理模組錯誤
```
無法讀取影片: synced_input_side_tennis_side.MP4
division by zero
```

## 🛠️ 解決方案

### 1. FFmpeg 安裝
程式已更新為自動檢測 FFmpeg 可用性：

#### 自動安裝（推薦）
```bash
# 執行自動安裝腳本
install_ffmpeg.bat
```

#### 手動安裝
1. **使用 Chocolatey**（推薦）
   ```bash
   choco install ffmpeg -y
   ```

2. **使用 Winget**（Windows 10/11）
   ```bash
   winget install ffmpeg
   ```

3. **手動下載**
   - 訪問: https://ffmpeg.org/download.html
   - 下載 Windows 版本
   - 解壓縮到 `C:\ffmpeg`
   - 將 `C:\ffmpeg\bin` 添加到系統 PATH

### 2. 程式改進

#### 📁 檔案更新
- ✅ `trajector_processing_simple_test.py` - 新增 FFmpeg 檢查功能
- ✅ `trajector_processing_with_segmentation.py` - 增強錯誤處理
- ✅ `install_ffmpeg.bat` - FFmpeg 自動安裝腳本
- ✅ `quick_test_improved.bat` - 改進的測試腳本

#### 🔧 功能改進
1. **智能 FFmpeg 檢測**: 自動檢查並提示安裝
2. **優雅降級**: FFmpeg 不可用時跳過影片分割功能
3. **增強錯誤處理**: 防止程式因單一模組失敗而崩潰
4. **詳細日誌**: 更清楚的錯誤訊息和執行狀態

## 🚀 新的執行方式

### 方式1: 改進的批次檔（推薦）
```bash
quick_test_improved.bat
```

### 方式2: 直接執行 Python
```bash
python trajector_processing_simple_test.py
```

### 方式3: 完整安裝流程
```bash
# 1. 安裝 FFmpeg
install_ffmpeg.bat

# 2. 執行測試
quick_test_improved.bat
```

## ⚙️ 程式行為變化

### 🎯 FFmpeg 可用時
- ✅ 完整功能：影片同步 → 自動分割 → 軌跡分析 → GPT 反饋
- ✅ 組織化結果儲存

### 🎯 FFmpeg 不可用時
- ⚠️ 限制功能：影片同步 → 軌跡分析（使用完整影片）→ GPT 反饋
- ✅ 仍可產生分析結果，但無法分割影片片段

## 📊 測試結果預期

### 成功執行後，您將看到：
```
tennis_analysis_sessions/
├── tennis_analysis_YYYYMMDD_HHMMSS/
│   ├── 00_input_videos/        # 原始影片
│   ├── 01_synced_videos/       # 同步後影片
│   ├── 02_segmented_videos/    # 分割片段（需要FFmpeg）
│   ├── 03_2d_trajectories/     # 2D軌跡數據
│   ├── 04_processed_videos/    # 處理後影片
│   ├── 05_3d_trajectories/     # 3D軌跡數據
│   ├── 06_analysis_results/    # KNN分析結果
│   ├── 07_final_reports/       # GPT反饋報告
│   ├── logs/                   # 執行日誌
│   └── README.md              # 執行摘要
```

## 🔍 故障排除

### 如果仍然遇到問題：

1. **檢查 Python 環境**
   ```bash
   python --version
   pip list | findstr ultralytics
   ```

2. **檢查影片檔案**
   - 確認 `input_videos/` 中有正確的影片檔案
   - 支援格式：MP4, AVI, MOV

3. **檢查模型檔案**
   ```
   model/
   ├── tennisball_OD_v1.pt
   ├── yolov8n-pose.pt
   └── yolov8n.pt
   ```

4. **查看詳細日誌**
   - 檢查 `tennis_analysis_sessions/*/logs/` 中的錯誤日誌

## 💡 建議

1. **首次執行**: 建議先安裝 FFmpeg 以獲得完整功能
2. **測試環境**: 使用較短的測試影片進行初次驗證
3. **系統需求**: 確保有足夠的硬碟空間（建議 > 5GB）
4. **GPU 加速**: 如有 NVIDIA GPU，確保 CUDA 環境正確設定

## 📞 技術支援

如果問題持續存在，請提供：
1. 錯誤訊息截圖
2. `logs/` 資料夾中的錯誤日誌
3. 系統環境資訊（Windows 版本、Python 版本）