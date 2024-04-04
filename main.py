import os

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# token
TOKEN = "TOKEN"

def start(update: Update, _: CallbackContext) -> None:
    # send a message when the '/start' command is sent
    update.message.reply_text("I'm a bot!")

def hello(update: Update, _: CallbackContext) -> None:
    update.message.reply_text(f"Hello {update.effective_user.first_name}")

def main() -> None:
    updater = Updater(token=TOKEN)

    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("hello", hello))

    # start the Bot
    updater.start_polling()

    # run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()



