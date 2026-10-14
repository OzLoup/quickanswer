import webbrowser

import uvicorn

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8765")
    uvicorn.run("quickanswer.server:app", host="127.0.0.1", port=8765, log_level="warning")
