import shutil
import os

def flatten_data(data_path="./0426_EN_used_task2", dest_path="./covid19_data_0426_flatten"):
    if os.path.exists(dest_path):
        print("Skipping flattening data as the dest_path already exists")
        return
    os.mkdir(dest_path)
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for date in os.listdir(folder_path):
            date_path = os.path.join(folder_path, date)
            for audio in os.listdir(date_path):
                audio_path = os.path.join(date_path, audio)
                new_file_name = "___".join([folder, date, audio])
                print("Copying %s to %s..." % (audio_path, os.path.join(dest_path, new_file_name)))
                shutil.copy2(audio_path, os.path.join(dest_path, new_file_name))

def main():
    flatten_data()

if __name__ == "__main__":
    main()
