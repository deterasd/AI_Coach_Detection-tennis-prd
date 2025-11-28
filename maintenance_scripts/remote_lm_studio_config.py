# LM Studio 遠程配置
# 當您使用 ngrok 或其他工具將 LM Studio API 分享到外網時使用

# 遠程 LM Studio API 設定
REMOTE_LM_STUDIO_URL = "https://labradoritic-sporular-evelyne.ngrok-free.dev/v1"  # 新的 ngrok URL
REMOTE_LM_STUDIO_API_KEY = "lm-studio"  # 如果 ngrok 需要驗證，請填入金鑰

# 使用說明：
# 1. 在另一台電腦啟動 LM Studio 並載入模型
# 2. 使用 ngrok 將本地端口分享到外網：
#    ngrok http 1234
# 3. 將生成的 ngrok URL 填入上面的 REMOTE_LM_STUDIO_URL
# 4. 如果 ngrok 需要驗證，填入對應的 API 金鑰
# 5. 儲存此檔案並重新運行程式