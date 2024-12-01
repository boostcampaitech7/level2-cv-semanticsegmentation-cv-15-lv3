import requests
from config import WEBHOOK_URL


# Discord Webhook URL
webhook_url = WEBHOOK_URL
def send_discord_message(content):
    """Discord Webhook으로 메시지 전송"""
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print("메시지 전송 실패:", response.status_code)