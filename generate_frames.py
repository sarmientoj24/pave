import pandas as pd
from subprocess import run
import os
from datetime import datetime, timedelta, timezone
from gpxcsv import gpxtolist
import cv2
import geopy.distance as dist


def generate_gpx(path: str):
    base_no_ext = os.path.splitext(path)[0]
    bin_outfile = f"{base_no_ext}.bin"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-codec",
        "copy",
        "-map",
        "0:3",
        "-f",
        "rawvideo",
        bin_outfile,
    ]
    run_proc = run(cmd, capture_output=True, text=True)
    run_proc.check_returncode()

    gpx_outfile = base_no_ext

    cmd = ["gopro2gpx", "-s", "-vvv", "-b", bin_outfile, gpx_outfile]
    run_proc = run(cmd, capture_output=True, text=True)
    run_proc.check_returncode()

    return bin_outfile, f"{gpx_outfile}.gpx"


def convert_str_to_ts(str_time):
    timeformat = "%Y-%m-%dT%H:%M:%S.%fZ"
    dt = datetime.strptime(str_time, timeformat).replace(
        tzinfo=(timezone(timedelta(0)))
    )
    millsec = int(dt.timestamp() * 1000)
    return millsec


def make_ts(row, first_ts):
    str_timestamp = str(row["time"])
    msec = convert_str_to_ts(str_timestamp)

    curr_ts = msec - first_ts
    row["ts"] = msec
    row["timestamp"] = curr_ts
    return row


def generate_timestamp(df):
    first_datetime = str(df.time[0])
    first_msec = convert_str_to_ts(first_datetime)

    df = df.apply(make_ts, axis=1, args=[first_msec])
    return df


def extract_frames_from_df(
    df: pd.DataFrame, video: str, output_dir: str, interval: int = 5
):
    data_time = df.time.tolist()
    lats = df["lat"].tolist()
    long = df["lon"].tolist()
    millis = df["timestamp"].tolist()

    data = list(zip(lats, long, data_time))
    time_frames = []
    indices = []

    cnt = 0
    i = 0

    while i < len(data):
        lat, long, data_time = data[i]
        j = i + 1

        while j < len(data):
            n_lat, n_long, n_data_time = data[j]
            mdist = dist.distance((lat, long), (n_lat, n_long)).km * 1000

            if mdist >= interval:
                time_frames.append(n_data_time)
                indices.append(j)
                cnt += 1
                break
            j += 1
        i = j

    # Get milliseconds
    ms = [millis[i] for i in indices]

    # Save frames
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()

    frame_count = 0
    while success:
        if frame_count % 50 == 0:
            print(frame_count)
        if frame_count == len(time_frames):
            break
        time_frame = time_frames[frame_count]
        vidcap.set(cv2.CAP_PROP_POS_MSEC, ms[frame_count])  # added this line
        success, image = vidcap.read()
        if success:
            filename = str(frame_count).zfill(10)
            cv2.imwrite(os.path.join(output_dir, f"{filename}.jpg"), image)
        else:
            print(f"Error at frame {time_frame}!")
            success = True
        frame_count += 1


def generate_frames(path: str, output_dir: str, interval: int = 5):
    bin_file, gpx_outfile = generate_gpx(path)
    gpx_file = gpxtolist(gpx_outfile)

    df = pd.DataFrame(gpx_file)
    df = generate_timestamp(df)

    # Extract frames
    extract_frames_from_df(df, path, output_dir, interval)

    # Delete files
    print("Deleting bin, gpx, and MP4 files. . .")
    os.remove(bin_file)
    os.remove(gpx_outfile)
    os.remove(path)
