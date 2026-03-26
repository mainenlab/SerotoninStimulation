# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 03/03/2026
"""
# %%

from pathlib import Path
import shutil

LOCAL_PATH = Path(r'F:\Guido\Subjects')
SERVER_PATH = Path(r'V:\imaging1\guido\Subjects')

# Loop over subjects
for subject_path in LOCAL_PATH.iterdir():
    if not subject_path.is_dir():
        continue
    # Loop over sessions
    for session_path in subject_path.iterdir():
        if not session_path.is_dir():
            continue
        # Check if flag file exists on server
        server_session_path = SERVER_PATH.joinpath(subject_path.name, session_path.name)
        if server_session_path.joinpath('eyetrack_me_fr.flag').exists():
            # Copy raw video data
            local_video_path = session_path.joinpath('raw_video_data')
            server_video_path = server_session_path.joinpath('raw_video_data')
            if local_video_path.exists():
                server_video_path.mkdir(exist_ok=True)
                for video_file in local_video_path.iterdir():
                    if video_file.is_file() and not server_video_path.joinpath(video_file.name).exists():
                        print(f'Copying {video_file.name} to {server_video_path}')
                        shutil.copy(video_file, server_video_path)