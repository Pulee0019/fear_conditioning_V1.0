# logger.py
import tkinter as tk
from datetime import datetime


log_text_widget = None


def set_log_widget(widget):
    """Set the Text widget to be used for logging messages."""
    global log_text_widget
    log_text_widget = widget


def log_message(message, level="INFO"):
    """Log a message with the specified level (INFO, WARNING, ERROR) to the Text widget if set, or print to console otherwise."""
    if log_text_widget:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "ERROR":
            tag = "error"
            prefix = f"[{timestamp}] ERROR: "
        elif level == "WARNING":
            tag = "warning"
            prefix = f"[{timestamp}] WARNING: "
        else:
            tag = "info"
            prefix = f"[{timestamp}] INFO: "
        
        log_text_widget.tag_config("error", foreground="red")
        log_text_widget.tag_config("warning", foreground="orange")
        log_text_widget.tag_config("info", foreground="black")

        log_text_widget.config(state="normal")
        log_text_widget.insert(tk.END, prefix + message + "\n", tag)
        log_text_widget.see(tk.END)
        log_text_widget.config(state="disabled")
    else:
        print(f"[{level}] {message}")