import os
import sys

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == '__main__':
    demo = create_gradio_ui()
    print("\n Launching RAG Assistant...")
    demo.launch(css=custom_css)