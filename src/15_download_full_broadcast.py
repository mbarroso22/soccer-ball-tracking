import os
from SoccerNet.Downloader import SoccerNetDownloader

# Download location. This folder should be in .gitignore.
LOCAL_DIR = "SoccerTrackingData/SoccerNetFull"

# Read password from environment variable
PASSWORD = os.getenv("SOCCERNET_PASSWORD")

if PASSWORD is None:
    raise ValueError(
        "Missing password. Set it first with:\n"
        'PowerShell: $env:SOCCERNET_PASSWORD="your_password_here"'
    )

downloader = SoccerNetDownloader(LocalDirectory=LOCAL_DIR)
downloader.password = PASSWORD

# Start small: download low-res 224p videos first
# Files are 1st half and 2nd half videos.
downloader.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"],
    split=["train"]
)

print("Download complete.")