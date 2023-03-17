import webbrowser
import os

count_file = "count.txt"
google_url = "https://www.google.com/"

def open_google():
    webbrowser.open(google_url)

def get_count():
    if os.path.exists(count_file):
        with open(count_file, "r") as f:
            return int(f.read())
    else:
        return 0

def save_count(count):
    with open(count_file, "w") as f:
        f.write(str(count))

count = get_count()
if count >= 50:
    os.remove(count_file)
    os.system("""osascript -e 'tell app "System Events" to display dialog "已经打开50次" buttons "OK"'""")
else:
    open_google()
    save_count(count + 1)
