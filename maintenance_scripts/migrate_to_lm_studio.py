"""
自動遷移腳本：將專案從 OpenAI GPT 切換到 LM Studio
使用方法：python migrate_to_lm_studio.py
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """備份檔案"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ 已備份: {os.path.basename(filepath)} -> {os.path.basename(backup_path)}")
        return True
    return False

def modify_trajectory_gpt():
    """修改 trajectory_gpt.py"""
    filepath = "trajectory_gpt.py"
    
    if not os.path.exists(filepath):
        print(f"⚠️  找不到 {filepath}，跳過")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否已經修改過
    if 'from ai_config import ai_config' in content:
        print(f"✅ {filepath} 已經使用 ai_config，無需修改")
        return
    
    # 替換 import
    content = content.replace(
        'from openai import OpenAI\nimport pandas as pd \nimport single_feedback.prompt as prompt, single_feedback.model_config as model_config\nfrom open_ai_key import api_key',
        'from openai import OpenAI\nimport pandas as pd \nimport single_feedback.prompt as prompt, single_feedback.model_config as model_config\nfrom ai_config import ai_config'
    )
    
    # 替換 client 初始化
    content = content.replace(
        'self.client = OpenAI(api_key=api_key)',
        'self.client = ai_config.get_client()'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已修改: {filepath}")

def modify_trajectory_gpt_overall_feedback():
    """修改 trajectory_gpt_overall_feedback.py"""
    filepath = "trajectory_gpt_overall_feedback.py"
    
    if not os.path.exists(filepath):
        print(f"⚠️  找不到 {filepath}，跳過")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否已經修改過
    if 'from ai_config import ai_config' in content:
        print(f"✅ {filepath} 已經使用 ai_config，無需修改")
        return
    
    # 替換 import
    content = content.replace(
        'from open_ai_key import api_key\n\n# --- 全域設定 ---\nclient = OpenAI(api_key=api_key)',
        'from ai_config import ai_config\n\n# --- 全域設定 ---\nclient = ai_config.get_client()'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已修改: {filepath}")

def check_required_files():
    """檢查必要檔案是否存在"""
    print("\n=== 檢查必要檔案 ===")
    
    required_files = {
        'ai_config.py': '自動切換管理器',
        'remote_lm_studio_config.py': '遠程 LM Studio 配置',
        'single_feedback/model_config.py': '模型配置'
    }
    
    missing_files = []
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            print(f"✅ {filepath} ({description})")
        else:
            print(f"❌ 缺少: {filepath} ({description})")
            missing_files.append(filepath)
    
    return len(missing_files) == 0

def show_next_steps():
    """顯示後續步驟"""
    print("\n" + "="*60)
    print("🎉 遷移完成！")
    print("="*60)
    print("\n📋 後續步驟：")
    print("\n1️⃣ 啟動 LM Studio")
    print("   - 載入模型（例如：google/gemma-3n-e4b）")
    print("   - 啟動 Local Server (預設 port 1234)")
    
    print("\n2️⃣ 啟動 ngrok")
    print("   - 執行: ngrok http 1234")
    print("   - 複製 ngrok 提供的網址")
    
    print("\n3️⃣ 更新配置")
    print("   - 編輯 remote_lm_studio_config.py")
    print("   - 將 REMOTE_LM_STUDIO_URL 改成您的 ngrok 網址")
    print("   - 格式: https://your-url.ngrok-free.dev/v1")
    
    print("\n4️⃣ 驗證設定")
    print("   - 執行: python test_lm_studio_feedback.py")
    print("   - 確認看到 ✅ 已連接到 LM Studio 服務器")
    
    print("\n5️⃣ 測試完整流程")
    print("   - 執行: python test_gpt_feedback_quick.py")
    print("   - 確認回饋生成正常")
    
    print("\n💡 提示：")
    print("   - 如果 LM Studio 無法連接，系統會自動切換到 OpenAI API")
    print("   - 所有原始檔案都已備份（.backup_* 檔案）")
    print("   - 如需還原，請刪除修改後的檔案，並移除 .backup 副檔名")
    print("\n" + "="*60)

def main():
    print("="*60)
    print("🔄 開始遷移專案到 LM Studio")
    print("="*60)
    
    # 檢查必要檔案
    if not check_required_files():
        print("\n❌ 缺少必要檔案，請先確保以下檔案存在：")
        print("   - ai_config.py")
        print("   - remote_lm_studio_config.py")
        print("   - single_feedback/model_config.py")
        return
    
    print("\n=== 開始修改檔案 ===")
    
    # 修改各個檔案
    modify_trajectory_gpt()
    modify_trajectory_gpt_overall_feedback()
    
    # 檢查 trajectory_gpt_single_feedback.py
    if os.path.exists("trajectory_gpt_single_feedback.py"):
        with open("trajectory_gpt_single_feedback.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from ai_config import ai_config' in content:
                print("✅ trajectory_gpt_single_feedback.py 已經使用 ai_config")
            else:
                print("⚠️  trajectory_gpt_single_feedback.py 需要手動檢查")
    
    # 顯示後續步驟
    show_next_steps()

if __name__ == "__main__":
    main()
