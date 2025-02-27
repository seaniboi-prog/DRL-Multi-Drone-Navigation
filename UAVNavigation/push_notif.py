from pushbullet import Pushbullet

# Your Pushbullet API key

pb = Pushbullet(api_key)

# Define a function to send a notification
def send_notification(title, message):
    push = pb.push_note(title, message)
    print(f"Notification sent! ID: {push['iden']}")
    
def send_notif_image(title, message, image_path):
    with open(image_path, "rb") as pic:
        file_data = pb.upload_file(pic, "image.jpg")
        push = pb.push_file(**file_data, title=title, body=message)
        print(f"Notification sent! ID: {push['iden']}")

if __name__ == "__main__":
    send_notification("Test", "This is a test message")
    send_notif_image("Test", "This is a test message", "plots/train/reward_train_plot_disc_cust_ppo_random_single.png")
