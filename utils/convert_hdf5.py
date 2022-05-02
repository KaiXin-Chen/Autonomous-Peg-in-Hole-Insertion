import os
import h5py
import pandas as pd
from tqdm import tqdm
import cv2

def convert_episode(data_folder, logs, idx):
    print(idx)
    format_time = logs.iloc[idx].Time.replace(":", "_")
    trial = os.path.join(data_folder, format_time)
    print(os.path.join(trial, "data.hdf5"))

    all_datasets = h5py.File(os.path.join(trial, "data.hdf5"), 'r')
    streams = ["cam_gripper_color", "cam_fixed_color"]
    for stream in streams:
        os.makedirs(os.path.join(trial, stream), exist_ok=True)
        frame_chunks = all_datasets[stream].iter_chunks()
        for frame_nb, frame_chunk in enumerate(tqdm(frame_chunks)):
            img = all_datasets[stream][frame_chunk]
            if not stream.endswith("flow"):
                out_file = os.path.join(trial, stream, str(frame_nb) + ".png")
                if not os.path.exists(out_file) or True:
                    cv2.imwrite(out_file, img)

if __name__ == "__main__":
    logs = pd.read_csv("data/data_0214/episode_times.csv")
    data_folder = "data/data_0214/test_recordings"
    for idx in range(len(logs)):
        convert_episode(data_folder, logs, idx)