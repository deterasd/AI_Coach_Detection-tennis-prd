# 球配對問題修復指南

## 🐛 問題描述

**症狀**: 雖然影片分割產生了 3 個側面片段和 3 個 45度片段，但球配對只產生 1 個球對，導致只處理第一顆球。

**根本原因**: `align_ball_segments` 函數中有重複的程式碼區塊（已修復），並且 Python 可能快取了舊版本的程式碼。

## ✅ 已修復的問題

### 1. 程式碼修復
- **檔案**: `trajector_processing_unified.py`
- **修復**: 刪除了 `align_ball_segments` 函數中的重複程式碼（第 1114-1210 行）
- **結果**: 函數現在會正確處理所有球對，而不是只處理第一個

### 2. 時間調整
- **修改**: 球出場時間間隔從 0.3s 縮短為 0.1s
- **影響**: 片段時間更精確，不會包含過多冗餘畫面

### 3. Python 快取清除
- **操作**: 清除 `__pycache__` 中的 `.pyc` 檔案
- **目的**: 確保系統載入最新版本的程式碼

## 🔧 修復工具

提供了 3 個工具來處理不同情況：

### 1. `check_tim82.py` - 診斷工具
```bash
python check_tim82.py
```
- 檢查球配對數量
- 顯示每個片段的詳細時間
- 列出已產生的球資料夾

### 2. `reprocess_ball_pairs.py` - 重新配對工具
```bash
python reprocess_ball_pairs.py
```
- 只重新配對球片段
- 更新 `segmentation_results.json`
- 產生備份檔案

### 3. `reprocess_trajectories.py` - 重新處理工具
```bash
python reprocess_trajectories.py
```
- 基於更新後的球對
- 重新執行完整的軌跡分析
- 產生 trajectory_1, trajectory_2, trajectory_3 資料夾

### 4. `fix_ball_processing.py` - 一鍵修復工具 ⭐
```bash
python fix_ball_processing.py
```
- **推薦使用**
- 自動執行步驟1和步驟2
- 一次完成所有修復

## 📊 修復前後對比

### 修復前 (TIM82 範例)
```
側面片段: 3 個
45度片段: 3 個
球對數量: 1 對  ❌
  球1: 側面0.23s ↔ 45度0.70s

球資料夾:
  trajectory_1/  ✅
  (缺少 trajectory_2 和 trajectory_3)  ❌
```

### 修復後
```
側面片段: 3 個
45度片段: 3 個
球對數量: 3 對  ✅
  球1: 側面0.23s ↔ 45度0.70s (差異0.47s)
  球2: 側面2.03s ↔ 45度2.80s (差異0.77s)
  球3: 側面3.83s ↔ 45度4.90s (差異1.07s)

球資料夾:
  trajectory_1/  ✅ (8 個 JSON 檔案)
  trajectory_2/  ✅ (8 個 JSON 檔案)
  trajectory_3/  ✅ (8 個 JSON 檔案)
```

## 🚀 使用流程

### 情況A: 已經有問題的使用者 (例如 TIM82)

```bash
# 方法1: 一鍵修復 (推薦)
python fix_ball_processing.py
# 輸入: TIM82

# 方法2: 分步執行
python reprocess_ball_pairs.py     # 步驟1: 重新配對
# 輸入: TIM82

python reprocess_trajectories.py   # 步驟2: 重新處理
# 輸入: TIM82
```

### 情況B: 新的使用者

```bash
# 直接執行完整流程
python trajector_processing_simple_test.py
```

確保：
1. 已清除 Python 快取 (`__pycache__` 資料夾)
2. 使用最新版本的 `trajector_processing_unified.py`

## 🔍 驗證修復

執行修復後，使用診斷工具確認：

```bash
python check_tim82.py
```

應該看到：
- ✅ 球對數量: 3 對
- ✅ 3 個球資料夾: trajectory_1, trajectory_2, trajectory_3
- ✅ 每個資料夾都有 8 個 JSON 檔案

## 📝 關鍵檔案說明

### `segmentation_results.json`
```json
{
  "side_segments": [...],      // 側面片段清單
  "deg45_segments": [...],     // 45度片段清單
  "ball_pairs": [              // 配對結果
    {
      "ball_number": 1,
      "side_data": {...},
      "deg45_data": {...},
      "time_difference": 0.47,
      "status": "paired"
    },
    // ... 球2, 球3
  ]
}
```

### 備份檔案
- `segmentation_results.json.backup` - 重新配對前的備份
- 如果修復失敗，可以還原此檔案

## ⚠️ 常見問題

### Q1: 為什麼只處理了第一顆球？
**A**: 舊版本的 `align_ball_segments` 函數有重複程式碼，導致函數提前 return。已在最新版本修復。

### Q2: 如何確保使用最新版本的程式碼？
**A**: 
1. 刪除 `__pycache__` 資料夾
2. 重新啟動 Python 或終端機
3. 使用 `importlib.reload()` 強制重新載入模組

### Q3: 重新配對會影響原始資料嗎？
**A**: 不會。工具會：
1. 自動建立備份檔案 (`.backup`)
2. 不修改原始影片片段
3. 只更新配對關係

### Q4: 需要重新分割影片嗎？
**A**: **不需要**。影片片段已經正確分割，只需要：
1. 重新配對球片段
2. 重新執行軌跡分析

## 🎯 預期結果

修復完成後，每個使用者應該有：

```
trajectory/(使用者名稱)__trajectory/
├── (使用者名稱)__1_side.mp4
├── (使用者名稱)__1_45.mp4
├── (使用者名稱)__segmentation_results.json  (3 個球對)
├── segments/
│   ├── (使用者名稱)__1_side_segment.mp4
│   ├── (使用者名稱)__2_side_segment.mp4
│   ├── (使用者名稱)__3_side_segment.mp4
│   ├── (使用者名稱)__1_45_segment.mp4
│   ├── (使用者名稱)__2_45_segment.mp4
│   └── (使用者名稱)__3_45_segment.mp4
├── trajectory_1/      ← 球1的完整分析
│   ├── ...8個JSON檔案
│   └── ...處理後的影片
├── trajectory_2/      ← 球2的完整分析
│   ├── ...8個JSON檔案
│   └── ...處理後的影片
└── trajectory_3/      ← 球3的完整分析
    ├── ...8個JSON檔案
    └── ...處理後的影片
```

## 📞 需要協助？

如果遇到問題：
1. 檢查控制台錯誤訊息
2. 使用 `check_tim82.py` 診斷當前狀態
3. 查看備份檔案是否存在
4. 確認 Python 快取已清除

---

**最後更新**: 2025-11-04
**版本**: 2.0
**狀態**: ✅ 已修復並測試
