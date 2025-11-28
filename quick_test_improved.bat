@echo off
title 🎾 網球 AI 教練 - 快速測試
echo 🎾 網球 AI 教練 - 快速測試
echo ============================================================
echo.

echo 🔍 檢查 FFmpeg 安裝狀態...
ffmpeg -version >nul 2>nul
if %errorlevel% == 0 (
    echo ✅ FFmpeg 已安裝並可使用
) else (
    echo ❌ FFmpeg 未安裝或不可用
    echo 📖 請執行 install_ffmpeg.bat 安裝 FFmpeg
    echo    或者程式將跳過影片分割功能
    echo.
    set /p continue="是否繼續執行？(y/n): "
    if /i not "%continue%"=="y" goto :end
)

echo.
echo 🐍 啟動 Python 分析程式...
python trajector_processing_simple_test.py

:end
echo.
echo 📁 測試完成！結果保存在 tennis_analysis_sessions/ 資料夾中
pause