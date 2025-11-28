#!/usr/bin/env python3
"""
從 USB 裝置信息中提取 GoPro 序號
"""
import subprocess
import re
from open_gopro import WiredGoPro
import asyncio

def extract_serial_from_usb():
    """從 USB 裝置信息中提取序號"""
    try:
        # 獲取詳細的 USB 裝置信息
        result = subprocess.run([
            'wmic', 'path', 'Win32_PnPEntity', 
            'where', "Name like '%HERO13%'", 
            'get', 'Name,DeviceID,HardwareID,InstanceId'
        ], capture_output=True, text=True, encoding='utf-8')
        
        print("USB 裝置信息:")
        print(result.stdout)
        
        # 嘗試從 DeviceID 中提取序號模式
        device_ids = re.findall(r'USB\\.*', result.stdout)
        for device_id in device_ids:
            print(f"裝置 ID: {device_id}")
            # GoPro 序號通常在 InstanceId 中
            
    except Exception as e:
        print(f"提取 USB 序號失敗: {e}")

async def try_common_serials():
    """嘗試常見的序號格式"""
    # 從 USB PID 0059 推斷，這可能是 HERO13
    # 常見的序號格式
    possible_serials = [
        # 基於 DeviceID 的可能序號
        "C35313D663AC3", 
        # 你程式中現有的序號
        "C3531324813253",
        "C3531350279436"
    ]
    
    print("嘗試常見序號格式...")
    for serial in possible_serials:
        try:
            print(f"正在嘗試序號: {serial}")
            gopro = WiredGoPro(serial=serial)
            await gopro.open()
            print(f"✅ 成功！你的 GoPro 序號是: {serial}")
            await gopro.close()
            return serial
        except Exception as e:
            print(f"❌ 序號 {serial} 失敗: {type(e).__name__}")
    
    return None

async def auto_detect_without_serial():
    """不指定序號的自動偵測"""
    try:
        print("嘗試自動偵測（無序號）...")
        gopro = WiredGoPro()  # 不指定序號
        await gopro.open()
        
        # 嘗試獲取裝置信息
        info = await gopro.http_command.get_camera_info()
        print(f"✅ 自動偵測成功!")
        print(f"相機信息: {info.data}")
        
        await gopro.close()
        return True
        
    except Exception as e:
        print(f"❌ 自動偵測失敗: {e}")
        return False

if __name__ == "__main__":
    print("=== GoPro 序號偵測工具 ===")
    print()
    
    print("1. 從 USB 裝置信息提取:")
    extract_serial_from_usb()
    print()
    
    print("2. 嘗試常見序號:")
    serial = asyncio.run(try_common_serials())
    print()
    
    if not serial:
        print("3. 嘗試自動偵測:")
        asyncio.run(auto_detect_without_serial())
    
    print()
    print("如果以上方法都失敗，請:")
    print("1. 確認 GoPro 已開機並連接 USB")
    print("2. 在 GoPro 上啟用 'Wired USB' 或 'USB Connection'")
    print("3. 連接時選擇 'Control Camera' 而不是 'Transfer Files'")
    print("4. 在 GoPro 設定中查看序號")