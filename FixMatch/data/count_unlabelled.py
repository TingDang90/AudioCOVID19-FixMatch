import os
import pickle

# load labels
with open("unlabelled_labels.pk", "rb") as f:
    all_data = pickle.load(f)

# count positive and negative samples
num_pos, num_neg = 0, 0
for uid in os.listdir("."):
    if "." in uid:
        continue
    uid_path = os.path.join(".", uid)
    for sample in os.listdir(uid_path):
        sample_path = os.path.join(uid, sample)
        if sample_path in all_data["pos"]:
            num_pos += 1
        elif sample_path in all_data["neg"]:
            num_neg += 1
        else:
            print("exception", sample_path)
            raise Exception

print(num_pos, num_neg)
