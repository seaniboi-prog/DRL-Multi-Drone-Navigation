import telegram
import asyncio
from datetime import datetime, timezone
import time

from telegram import Update
from typing import Optional

CONFIRMATION_LIST = ["Yes", "yes", "Y", "y"]
DECLINE_LIST = ["No", "no", "N", "n"]

BOT_API_KEY = '6523572576:AAENjdiohz8m0fSrT7ALDxNx3GcVEiqznNk'

bot = telegram.Bot(token=BOT_API_KEY)

CHANNEL_ID = '6690861694'

async def send_photo_async(photo_path, caption=None):
    return await bot.send_photo(chat_id=CHANNEL_ID, photo=open(photo_path, 'rb'), caption=caption) # type: ignore

async def send_message_async(message):
    return await bot.send_message(chat_id=CHANNEL_ID, text=message) # type: ignore

async def read_latest_chat_async():
    updates: list[Update] = await bot.get_updates() # type: ignore
    if len(updates) > 0:
        return updates[-1]
    else:
        return None
    
def wait_for_response(initial_message, timeout=10, interval=30):
    loop = asyncio.get_event_loop()
    timeout_seconds = timeout * 60
    interval_seconds = interval

    if initial_message:
        loop.run_until_complete(send_message_async(initial_message))
    loop_start_date = datetime.now(timezone.utc)

    print("Waiting for message...")
    
    while True:
        last_update: Optional[Update] = loop.run_until_complete(read_latest_chat_async())
        if last_update is not None and last_update.message is not None:
            latest_date = last_update.message.date
            if latest_date and latest_date > loop_start_date:
                if last_update.message.text in CONFIRMATION_LIST:
                    loop.run_until_complete(send_message_async("Received confirmation, continuing training..."))
                    return True
                elif last_update.message.text in DECLINE_LIST:
                    loop.run_until_complete(send_message_async("Received decline, stopping training..."))
                    return False
                else:
                    loop.run_until_complete(send_message_async("Received invalid message, try again..."))
            
        time.sleep(interval_seconds)
        timeout_seconds -= interval_seconds

        if timeout_seconds <= 0:
            loop.run_until_complete(send_message_async("No response received, stopping training..."))
            return False

def send_message(message):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_message_async(message))
    print("Message Sent")
    
def send_photo(photo_path, caption=None):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_photo_async(photo_path, caption))
    print(f"Photo: {photo_path} Sent")
    

# confirm = wait_for_response("Would you like to continue training?", timeout=1, interval=30)
# if confirm:
#     print("Confirmation Received")
# else:
#     print("No Message Received or Declined")

# send_photo("prog_rep_results.png", "Progress Report Results")