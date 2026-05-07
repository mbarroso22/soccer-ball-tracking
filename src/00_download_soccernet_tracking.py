from SoccerNet.Downloader import SoccerNetDownloader

LOCAL_DIR = "SoccerTrackingData/SoccerNet"

downloader = SoccerNetDownloader(LocalDirectory=LOCAL_DIR)

downloader.downloadDataTask(
    task="tracking",
    split=["train", "test"]
)