import requests
import pynvml
import time

# Discord Webhook URL
webhook_url = 'https://discord.com/api/webhooks/1305343891964428318/G85AIWjdio2VBY7V-egcaI-qJDOOcRAAVrThsUh6yYmKMdKT5Ff4HNkMkk8gWkkCNdWV'

def send_discord_message(content):
    """Discord Webhook으로 메시지 전송"""
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print("메시지 전송 실패:", response.status_code)
        

def monitor_gpu():
    """GPU 메모리 사용량을 모니터링하고 경고 메시지를 전송하는 함수"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    while True:
        # GPU 메모리 정보 가져오기
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory = info.used / info.total

        # 메모리 사용량이 50%를 넘으면 Discord로 경고 메시지 전송
        if used_memory > 0.9:
            message = f"🚨 [서버 3] 경고: GPU 메모리가 50% 이상 사용 중입니다! 현재 사용량: {used_memory * 100:.2f}%"
            send_discord_message(message)
        
        # 주기적으로 체크 (예: 10초 간격)
        time.sleep(10)