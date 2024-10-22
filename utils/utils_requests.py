import requests

def send_telegram_message(message):
    bot_token = '7440007022:AAFV2EObI3TzZ9w4vx-79iQNIjQgU-CTnds'
    chat_id = '150280965'
    send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}'
    response = requests.get(send_text)
    return response.json()


def send_start_training():
    message = '🌟✨ Training has started! Let’s crush those epochs! 💪😎'
    send_telegram_message(message)

def send_train_epoch_update(epoch, accuracy, loss):
    message = f'🔄 Epoch {epoch} completed!\nAccuracy: {accuracy:.2f}% 🎯\nLoss: {loss:.5f} 📉'
    send_telegram_message(message)

def send_validation_epoch_update(epoch, mAP, loss):
    message = f'🔍🔬 Validation Epoch {epoch}!\nmAP: {mAP:.2f}% 🎯\nLoss: {loss:.5f} 📉'
    send_telegram_message(message)

def send_training_complete(best_epoch, best_perf):
    message = f'🏁🏆 Training complete!\nBest epoch: {best_epoch}\nBest mAP: {best_perf:.2f}% 🎯'
    send_telegram_message(message)