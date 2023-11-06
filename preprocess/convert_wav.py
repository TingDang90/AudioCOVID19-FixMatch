import os

def main():
    """ Convert .m4a files to .wav files """
    data_path = "./unlabelled_data"
    users = os.listdir(data_path)
    for user in users:
        user_path = os.path.join(data_path, user)
        samples = os.listdir(user_path)
        for sample in samples:
            sample_path = os.path.join(user_path, sample)
            for audio_file in os.listdir(sample_path):
                audio_file = os.path.join(sample_path, audio_file)
                if audio_file.endswith(".m4a"):
                    output_file = audio_file.replace(".m4a", "-16k-mono.wav")
                    cmd = (
                        f"ffmpeg -i {audio_file} -ac 1 -ar 16000 -f wav {output_file}"
                    )
                    print(cmd)
                    os.system(cmd)
                    os.remove(audio_file)

if __name__ == "__main__":
    main()

