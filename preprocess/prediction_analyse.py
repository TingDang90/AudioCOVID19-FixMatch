import csv
import os
import shutil

def get_modality(filename):
    if 'cough' in filename:
        return 'c'
    if 'breath' in filename:
        return 'b'
    if 'voice' in filename or 'read' in filename:
        return 'v'
    print("No modality found for ", filename)
    return None


def find_pred_mismatches(csv_reader):
    num_mismatches = 0
    num_items = 0
    mismatches = {}
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        filename = row[0]
        pred_modality = row[6]
        modality = get_modality(filename)
        if modality != pred_modality:
            filename = row[0]
            user_date = "/".join(filename.split("/")[:-1])
            audio_file = filename.split("/")[-1]
            if user_date not in mismatches.keys():
                mismatches[user_date] = []
            mismatches[user_date].append(audio_file)
            num_mismatches += 1
        num_items += 1
    #print(mismatches)
    print("Mismatches: ", num_mismatches, "Items: ", num_items)
    print("This affects ", len(mismatches.keys()), " samples Out of ", num_items / 3)
    return mismatches


def main():
    with open("data_0426_yamnet_all_final.list") as f:
        csv_reader = csv.reader(f, delimiter=";")
        mismatches = find_pred_mismatches(csv_reader)
    for folder in mismatches.keys():
        print(os.path.join(os.getcwd(), "0426_EN_used_task2", folder))
        shutil.rmtree(os.path.join(os.getcwd(), "0426_EN_used_task2", folder))

    # remove empty dirs
    for folder in os.listdir(os.path.join(os.getcwd(), "0426_EN_used_task2")):
        user_path = os.path.join(os.getcwd(), "0426_EN_used_task2", folder)
        if len(list(os.listdir(user_path))) == 0:
            os.rmdir(user_path)

if __name__ == "__main__":
    main()

