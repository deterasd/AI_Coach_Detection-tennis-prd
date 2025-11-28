#!/usr/bin/env python3
"""
自動偵測 GoPro 序號
"""
from open_gopro import WiredGoPro
import asyncio

async def detect_gopro():
    try:
        print("正在自動偵測連接的 GoPro...")
        # 不指定序號，讓它自動偵測
        gopro = WiredGoPro()
        await gopro.open()
        
        print("✅ 成功連接到 GoPro！")
        
        # 獲取序號
        info = await gopro.http_command.get_camera_info()
        print(f"序號: {info.data}")
        
        await gopro.close()
        return True
        
    except Exception as e:
        print(f"❌ 自動偵測失敗: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(detect_gopro())