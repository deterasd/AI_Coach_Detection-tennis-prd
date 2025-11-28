"""
最簡遷移驗證腳本
只需要 ai_config.py 和 remote_lm_studio_config.py 就能運行
"""

def quick_migration_test():
    """快速測試遷移是否成功"""
    print("🧪 快速遷移測試")
    print("=" * 30)
    
    try:
        # 測試 ai_config 導入
        from ai_config import ai_config
        print("✅ ai_config.py 導入成功")
        
        # 測試客戶端初始化
        client = ai_config.get_client()
        if client:
            print("✅ AI 客戶端初始化成功")
            
            # 測試模型名稱獲取
            model_name = ai_config.get_model_name()
            print(f"✅ 模型名稱: {model_name}")
            
            # 測試提供者識別
            if ai_config.is_lm_studio():
                print("✅ 使用 LM Studio")
            else:
                print("✅ 使用 OpenAI API (自動回退)")
            
            # 簡單 API 測試
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "測試"}],
                    max_tokens=10
                )
                print("✅ API 呼叫測試成功")
                return True
            except Exception as api_error:
                print(f"⚠️ API 測試失敗: {api_error}")
                return False
                
        else:
            print("❌ AI 客戶端初始化失敗")
            return False
            
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        print("💡 請確認已複製 ai_config.py 到專案根目錄")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def check_required_files():
    """檢查必要檔案是否存在"""
    import os
    
    print("📁 檢查必要檔案...")
    
    required_files = {
        'ai_config.py': 'AI 配置管理核心',
        'remote_lm_studio_config.py': 'LM Studio 遠程配置'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename} ({description})")
        else:
            print(f"❌ {filename} ({description}) - 缺少")
            missing_files.append(filename)
    
    return len(missing_files) == 0

def show_minimal_setup_guide():
    """顯示最簡設置指南"""
    print("\n" + "=" * 50)
    print("🚀 最簡遷移指南")
    print("=" * 50)
    print("\n📋 只需要做這些：")
    print("\n1️⃣ 複製檔案到原始專案：")
    print("   - ai_config.py")
    print("   - remote_lm_studio_config.py")
    
    print("\n2️⃣ 修改一行程式碼：")
    print("   在 trajectory_gpt_single_feedback.py 中：")
    print("   MODEL = model_config.MODEL")
    print("   改成：")
    print("   MODEL = ai_config.get_model_name()")
    
    print("\n3️⃣ 設置 LM Studio：")
    print("   - 啟動 LM Studio + 載入模型")
    print("   - 啟動 Local Server")
    print("   - 更新 remote_lm_studio_config.py 中的網址")
    
    print("\n✅ 完成！系統會自動：")
    print("   - 創建 config.json (如果不存在)")
    print("   - 偵測 LM Studio 連接")
    print("   - 失敗時自動切換到 GPT")

def main():
    """主函式"""
    print("🎯 最簡 GPT → LM Studio 遷移測試")
    print("=" * 40)
    
    # 檢查檔案
    if not check_required_files():
        print("\n❌ 缺少必要檔案")
        show_minimal_setup_guide()
        return
    
    # 執行測試
    success = quick_migration_test()
    
    if success:
        print("\n🎉 遷移測試通過！")
        print("💡 您的專案已成功支援 LM Studio")
    else:
        print("\n⚠️ 需要檢查設置")
        show_minimal_setup_guide()

if __name__ == "__main__":
    main()