#!/usr/bin/env python3
"""
檢測連接的 GoPro 相機腳本
"""
import sys
import traceback

print(f"Python 版本: {sys.version}")
print("檢查依賴套件...")

try:
    from open_gopro import WiredGoPro
    print("✅ open_gopro 套件載入成功")
except ImportError as e:
    print(f"❌ open_gopro 套件載入失敗: {e}")
    exit(1)

try:
    import asyncio
    print("✅ asyncio 套件載入成功")
except ImportError as e:
    print(f"❌ asyncio 套件載入失敗: {e}")
    exit(1)

# 兩台相機的序號
CAMERA_SERIALS = {
    "Camera 1": "C3531324813253",
    "Camera 2": "C3531350279436"
}

async def test_camera_connection(camera_name, serial_number):
    """測試單一相機連接"""
    try:
        print(f"正在測試 {camera_name} (序號: {serial_number})...")
        gopro = WiredGoPro(serial=serial_number)
        print(f"正在嘗試打開連接...")
        await gopro.open()
        print(f"✅ {camera_name} 連接成功！")
        await gopro.close()
        return True
    except Exception as e:
        print(f"❌ {camera_name} 連接失敗:")
        print(f"   錯誤類型: {type(e).__name__}")
        print(f"   錯誤訊息: {str(e)}")
        print(f"   詳細錯誤:")
        traceback.print_exc()
        print()
        return False

async def main():
    print("開始檢測 GoPro 相機連接...")
    print("=" * 50)
    
    connected_cameras = []
    
    for camera_name, serial in CAMERA_SERIALS.items():
        success = await test_camera_connection(camera_name, serial)
        if success:
            connected_cameras.append((camera_name, serial))
    
    print("=" * 50)
    print("檢測結果：")
    
    if connected_cameras:
        print(f"找到 {len(connected_cameras)} 台連接的相機：")
        for camera_name, serial in connected_cameras:
            print(f"- {camera_name}: {serial}")
    else:
        print("未找到任何連接的相機")
        print("請確認：")
        print("1. 相機已開機")
        print("2. USB 線已正確連接")
        print("3. 相機未被其他程式佔用")

if __name__ == "__main__":
    asyncio.run(main())