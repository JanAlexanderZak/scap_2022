""" Utility functions.
"""

import datetime as dt

import matplotlib.pyplot as plt


def log_msg(string: str) -> str:
    """ Prints out a [LOG] message with a current timestamp.

    Args:
        string (str): String to be printed out.
    """
    ISO_TIME = dt.datetime.now().replace(microsecond=0).isoformat()
    string_msg = f"[LOG {ISO_TIME}] {string}"
    print(string_msg)
    return string_msg


def plot_loss(history):
  plt.plot(history.history["loss"], label="loss")
  plt.plot(history.history["val_loss"], label="val_loss")
  plt.xlabel("Epoch")
  plt.ylabel("Error [MPG]")
  plt.legend()
  plt.grid(True)