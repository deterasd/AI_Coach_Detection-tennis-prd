"""
AI 模型快速切換工具
使用方法：python switch_model.py [模型名稱]
支援模型：gpt-4o, gpt-4o-mini, lm-studio, auto
"""

import json
import os
from datetime import datetime

def load_config():
    """載入現有配置"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 如果檔案不存在，返回基本配置
        return {
            "ball_direction": "right",
            "confidence_threshold": 0.5,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
    except json.JSONDecodeError as e:
        print(f"❌ config.json 格式錯誤: {e}")
        return None

def save_config(config):
    """保存配置到檔案"""
    try:
        config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ 保存配置失敗: {e}")
        return False

def update_ai_model_config(provider, model_name="auto", **kwargs):
    """更新 AI 模型配置"""
    config = load_config()
    if config is None:
        return False
    
    # 確保 ai_model 區塊存在
    if "ai_model" not in config:
        config["ai_model"] = {}
    
    # 更新配置
    config["ai_model"]["provider"] = provider
    config["ai_model"]["model_name"] = model_name
    
    # 更新其他參數
    for key, value in kwargs.items():
        config["ai_model"][key] = value
    
    # 保存並返回結果
    return save_config(config)

def switch_to_gpt4o():
    """切換到 GPT-4o"""
    success = update_ai_model_config(
        provider="gpt-4o",
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=150
    )
    if success:
        print("✅ 已切換到 GPT-4o")
        print("   - 高品質回應，適合複雜分析")
        print("   - 成本較高，回應速度中等")
    return success

def switch_to_gpt4o_mini():
    """切換到 GPT-4o mini"""
    success = update_ai_model_config(
        provider="gpt-4o-mini",
        model_name="gpt-4o-mini",
        temperature=0.5,
        max_tokens=100
    )
    if success:
        print("✅ 已切換到 GPT-4o-mini")
        print("   - 快速回應，成本較低")
        print("   - 適合日常回饋分析")
    return success

def switch_to_lm_studio():
    """切換到 LM Studio"""
    success = update_ai_model_config(
        provider="lm-studio",
        model_name="auto",
        temperature=0.5,
        max_tokens=100
    )
    if success:
        print("✅ 已切換到 LM Studio")
        print("   - 本地運算，完全免費")
        print("   - 需要先啟動 LM Studio 和 ngrok")
    return success

def switch_to_auto():
    """自動偵測模式"""
    success = update_ai_model_config(
        provider="auto",
        model_name="auto",
        fallback_enabled=True,
        fallback_order=["lm-studio", "gpt-4o-mini", "gpt-4o"]
    )
    if success:
        print("✅ 已切換到自動偵測模式")
        print("   - 優先使用 LM Studio")
        print("   - 如果失敗會自動切換到 GPT")
    return success

def show_current_config():
    """顯示目前配置"""
    config = load_config()
    if config is None:
        return
    
    ai_config = config.get("ai_model", {})
    provider = ai_config.get("provider", "未設定")
    model_name = ai_config.get("model_name", "未設定")
    
    print(f"\n📊 目前 AI 模型配置:")
    print(f"   提供者: {provider}")
    print(f"   模型名稱: {model_name}")
    
    if "temperature" in ai_config:
        print(f"   溫度: {ai_config['temperature']}")
    if "max_tokens" in ai_config:
        print(f"   最大 tokens: {ai_config['max_tokens']}")
    
    print(f"   最後更新: {config.get('last_updated', '未知')}")

def show_help():
    """顯示使用說明"""
    print("🤖 AI 模型切換工具")
    print("=" * 50)
    print("\n使用方法:")
    print("  python switch_model.py [模型名稱]")
    print("\n支援的模型:")
    print("  gpt-4o        - OpenAI GPT-4o (高品質)")
    print("  gpt-4o-mini   - OpenAI GPT-4o-mini (快速)")  
    print("  lm-studio     - LM Studio 本地模型")
    print("  auto          - 自動偵測模式")
    print("  status        - 顯示目前配置")
    print("\n範例:")
    print("  python switch_model.py gpt-4o-mini")
    print("  python switch_model.py lm-studio")
    print("  python switch_model.py status")

def test_current_setup():
    """測試目前設定"""
    print("\n🧪 測試目前 AI 設定...")
    
    try:
        # 重新載入 ai_config
        import importlib
        import sys
        if 'ai_config' in sys.modules:
            importlib.reload(sys.modules['ai_config'])
        
        from ai_config import ai_config
        
        client = ai_config.get_client()
        if client:
            if ai_config.is_lm_studio():
                print("✅ LM Studio 連接成功")
            else:
                print("✅ OpenAI API 連接成功")
            
            # 嘗試簡單的 API 呼叫測試
            try:
                completion = client.chat.completions.create(
                    model=ai_config.current_model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                print("✅ API 呼叫測試成功")
            except Exception as e:
                print(f"⚠️ API 呼叫測試失敗: {e}")
        else:
            print("❌ 無法連接到任何 AI 服務")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def main():
    """主函式"""
    import sys
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # 處理各種命令
    if command in ["gpt-4o", "gpt4o"]:
        if switch_to_gpt4o():
            test_current_setup()
            
    elif command in ["gpt-4o-mini", "gpt4o-mini", "mini"]:
        if switch_to_gpt4o_mini():
            test_current_setup()
            
    elif command in ["lm-studio", "lm", "local"]:
        if switch_to_lm_studio():
            test_current_setup()
            
    elif command in ["auto", "automatic"]:
        if switch_to_auto():
            test_current_setup()
            
    elif command in ["status", "show", "current"]:
        show_current_config()
        test_current_setup()
        
    elif command in ["help", "-h", "--help"]:
        show_help()
        
    else:
        print(f"❌ 不支援的命令: {command}")
        print("\n支援的命令: gpt-4o, gpt-4o-mini, lm-studio, auto, status")
        show_help()

if __name__ == "__main__":
    main()