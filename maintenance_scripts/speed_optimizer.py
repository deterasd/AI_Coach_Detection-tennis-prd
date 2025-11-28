"""
AI 速度優化設定指南
提供不同場景的最佳配置參數
"""

import json
from datetime import datetime

# 預設配置組合
SPEED_CONFIGS = {
    "ultra_fast": {
        "name": "極速模式",
        "description": "最快速度，適合快速測試",
        "config": {
            "temperature": 0.1,     # 極低溫度，最確定性
            "max_tokens": 30,       # 極少 token，只輸出關鍵建議
            "top_p": 0.7,          # 限制候選詞彙
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "expected_speed": "0.5-1 秒"
    },
    
    "fast": {
        "name": "快速模式",
        "description": "平衡速度和品質",
        "config": {
            "temperature": 0.3,     # 低溫度，較確定
            "max_tokens": 50,       # 短回應
            "top_p": 0.8,          # 適中的詞彙選擇
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "expected_speed": "1-1.5 秒"
    },
    
    "balanced": {
        "name": "平衡模式",
        "description": "速度與品質兼顧",
        "config": {
            "temperature": 0.5,     # 中等溫度
            "max_tokens": 80,       # 中等長度
            "top_p": 0.9,          # 較多詞彙選擇
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "expected_speed": "1.5-2 秒"
    },
    
    "quality": {
        "name": "品質模式",
        "description": "重視回應品質，速度較慢",
        "config": {
            "temperature": 0.7,     # 較高溫度，更有創意
            "max_tokens": 120,      # 較長回應
            "top_p": 0.95,         # 更多詞彙選擇
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        },
        "expected_speed": "2-3 秒"
    }
}

def apply_speed_config(mode="fast"):
    """應用速度配置到 config.json"""
    if mode not in SPEED_CONFIGS:
        print(f"❌ 不支援的模式: {mode}")
        print(f"支援的模式: {', '.join(SPEED_CONFIGS.keys())}")
        return False
    
    speed_config = SPEED_CONFIGS[mode]
    
    try:
        # 讀取現有配置
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新 AI 模型配置
        if "ai_model" not in config:
            config["ai_model"] = {}
        
        # 應用速度配置
        config["ai_model"].update(speed_config["config"])
        config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # 保存配置
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已應用 {speed_config['name']}")
        print(f"📝 {speed_config['description']}")
        print(f"⏱️ 預期速度: {speed_config['expected_speed']}")
        print("\n📋 配置參數:")
        for key, value in speed_config["config"].items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置失敗: {e}")
        return False

def show_speed_tips():
    """顯示速度優化技巧"""
    print("🚀 AI 速度優化技巧")
    print("=" * 50)
    print("\n📊 參數說明:")
    print("   • temperature: 0.1-0.3 (低) = 快速但較死板")
    print("   •              0.4-0.6 (中) = 平衡速度與創意")
    print("   •              0.7-1.0 (高) = 慢但更有創意")
    print("")
    print("   • max_tokens:  30-50 (低) = 極速，簡短回應")
    print("   •              60-100 (中) = 平衡長度")
    print("   •              120+ (高) = 詳細但慢")
    print("")
    print("   • top_p:       0.6-0.8 (低) = 快速，較確定")
    print("   •              0.8-0.9 (中) = 平衡")
    print("   •              0.9-1.0 (高) = 慢，更多樣性")
    
    print("\n🎯 針對不同模型的建議:")
    print("   LM Studio (本地):")
    print("   - 小模型 (7B): temperature=0.2, max_tokens=40")
    print("   - 中模型 (13B): temperature=0.3, max_tokens=60")
    print("   - 大模型 (30B+): temperature=0.4, max_tokens=80")
    print("")
    print("   OpenAI API:")
    print("   - gpt-4o-mini: temperature=0.3, max_tokens=50")
    print("   - gpt-4o: temperature=0.5, max_tokens=80")
    
    print("\n💡 額外優化建議:")
    print("   1. 使用 LM Studio 比 OpenAI 更快 (本地運算)")
    print("   2. 選擇較小的模型 (7B 比 13B 快)")
    print("   3. 使用 GPU 加速 (CUDA/Metal)")
    print("   4. 關閉不必要的處理步驟")
    print("   5. 簡化提示詞內容")

def main():
    """主函式"""
    import sys
    
    if len(sys.argv) < 2:
        print("🚀 AI 速度優化工具")
        print("=" * 30)
        print("\n使用方法:")
        print("   python speed_optimizer.py [模式]")
        print("\n可用模式:")
        for key, config in SPEED_CONFIGS.items():
            print(f"   {key:<12} - {config['name']} ({config['expected_speed']})")
        print("   tips         - 顯示優化技巧")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "tips":
        show_speed_tips()
    elif mode in SPEED_CONFIGS:
        success = apply_speed_config(mode)
        if success:
            print("\n💡 提示:")
            print("   重新執行程式以套用新設定")
            print("   執行 python test_ai_switch.py 驗證速度")
    else:
        print(f"❌ 不支援的模式: {mode}")
        main()

if __name__ == "__main__":
    main()