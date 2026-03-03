# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 03/03/2026
"""
# %%
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SERVER_PATH = Path(r'V:\imaging1\guido\Subjects')
LOCAL_PATH = Path(r'F:\Guido\Subjects')

def sync_eyetrack_data(server_base_path: Path, local_base_path: Path):
    """
    Synchronizes 'raw_video_data' folders from local to server for sessions
    that have an 'eyetrack_me_fr.flag' file on the server.

    Args:
        server_base_path (Path): The base path on the server where subjects are located.
        local_base_path (Path): The base path locally where subjects are located.
    """
    if not server_base_path.is_dir():
        logging.error(f"Server base path does not exist or is not a directory: {server_base_path}")
        return
    if not local_base_path.is_dir():
        logging.error(f"Local base path does not exist or is not a directory: {local_base_path}")
        return

    logging.info(f"Starting synchronization from {local_base_path} to {server_base_path}")

    for subject_server_path in server_base_path.iterdir():
        if not subject_server_path.is_dir():
            continue # Skip non-directory items

        subject_name = subject_server_path.name
        subject_local_path = local_base_path / subject_name

        if not subject_local_path.is_dir():
            logging.warning(f"Skipping subject '{subject_name}': No corresponding local folder found at {subject_local_path}")
            continue

        logging.info(f"Processing subject: {subject_name}")

        for session_server_path in subject_server_path.iterdir():
            if not session_server_path.is_dir():
                continue # Skip non-directory items

            session_name = session_server_path.name
            flag_file_path = session_server_path / "eyetrack_me_fr.flag"

            if flag_file_path.is_file():
                logging.info(f"  Flag file found for session '{session_name}' in subject '{subject_name}'.")

                session_local_path = subject_local_path / session_name
                if not session_local_path.is_dir():
                    logging.warning(f"    Skipping session '{session_name}': No corresponding local session folder found at {session_local_path}")
                    continue

                local_raw_video_data_path = session_local_path / "raw_video_data"
                server_raw_video_data_path = session_server_path / "raw_video_data"

                if not local_raw_video_data_path.is_dir():
                    logging.warning(f"    Skipping session '{session_name}': Local 'raw_video_data' folder not found at {local_raw_video_data_path}")
                    continue

                logging.info(f"    Copying 'raw_video_data' from {local_raw_video_data_path} to {server_raw_video_data_path}")
                try:
                    # Use dirs_exist_ok=True to merge contents if destination exists.
                    # This requires Python 3.8+. For older versions, manual handling (e.g., deleting destination first) might be needed.
                    shutil.copytree(local_raw_video_data_path, server_raw_video_data_path, dirs_exist_ok=True)
                    logging.info(f"    Successfully copied 'raw_video_data' for session '{session_name}'.")
                except FileExistsError:
                    # This specific error should ideally be prevented by dirs_exist_ok=True for Python 3.8+
                    logging.error(f"    Destination folder '{server_raw_video_data_path}' already exists and could not be merged. Skipping copy.")
                except Exception as e:
                    logging.error(f"    Error copying 'raw_video_data' for session '{session_name}': {e}")
            else:
                logging.debug(f"  No flag file found for session '{session_name}' in subject '{subject_name}'. Skipping.")

    logging.info("Synchronization complete.")

if __name__ == "__main__":
    sync_eyetrack_data(SERVER_PATH, LOCAL_PATH)