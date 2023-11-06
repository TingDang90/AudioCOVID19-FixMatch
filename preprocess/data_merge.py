import os
import shutil
from pydub import AudioSegment

POS_DIR = "./unlabelled_positive"
NEG_DIR = "./unlabelled_negative"
DEST_DIR = "./merged_data"
MAX_SAMPLES_PER_USER = 5


def cut_samples(pos_samples, neg_samples):
    """ Fairly merge positive and negative samples up to MAX_SAMPLES_PER_USER """
    if len(pos_samples) + len(neg_samples) > MAX_SAMPLES_PER_USER:
        # cut down samples
        pos_samples.sort()
        neg_samples.sort()
        new_pos_len = max(MAX_SAMPLES_PER_USER - min(MAX_SAMPLES_PER_USER, len(neg_samples)), MAX_SAMPLES_PER_USER // 2)
        new_neg_len = max(MAX_SAMPLES_PER_USER - min(MAX_SAMPLES_PER_USER, len(pos_samples)), MAX_SAMPLES_PER_USER // 2)
        # if MAX_SAMPLES_PER_USER is odd, we miss a sample
        if new_pos_len + new_neg_len < MAX_SAMPLES_PER_USER:
            # gauge whether we can expand a pos or neg sample
            can_expand_pos, can_expand_neg = False, False
            if new_pos_len < len(pos_samples):
                can_expand_pos = True
            if new_neg_len < len(neg_samples):
                can_expand_neg = True
            # expand this sample
            if can_expand_pos and can_expand_neg:
                if random.random() < 0.5:
                    new_pos_len += 1
                else:
                    new_neg_len += 1
            elif can_expand_pos:
                new_pos_len += 1
            else:
                new_neg_len += 1
        # Incorrect length
        if len(pos_samples[:new_pos_len]) + len(neg_samples[:new_neg_len]) != MAX_SAMPLES_PER_USER:
            print(pos_samples[:new_pos_len], neg_samples[:new_neg_len])
            raise Exception
        return pos_samples[:new_pos_len], neg_samples[:new_neg_len]
    return pos_samples, neg_samples


def copy_files(src_dir, dest_dir):
    """ Copy files from src_dir to dest_dir """
    new_dir_name = src_dir.split(os.path.sep)[-1]
    user_name = src_dir.split(os.path.sep)[-2]
    new_user_dir = os.path.join(dest_dir, user_name)
    new_dir = os.path.join(new_user_dir, new_dir_name)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if not os.path.exists(new_user_dir):
        os.mkdir(new_user_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for item in os.listdir(src_dir):
        dest_full = os.path.join(new_dir, item)
        print("Copying from ", os.path.join(src_dir, item), " to ", dest_full)
        shutil.copy2(os.path.join(src_dir, item), dest_full)


def main():
    # check if data already exists
    if os.path.exists(DEST_DIR):
        print("Skipping data merging as destination dir already exists...")
        return
 
    # get sample names
    pos_samples = os.listdir(POS_DIR)
    neg_samples = os.listdir(NEG_DIR)
    
    # merge intersection
    intersection = set(pos_samples).intersection(set(neg_samples))
    if len(intersection) != 0:
        for user in intersection:
            print("Copying files for ", user)
            # get samples for this user
            pos_user_path = os.path.join(POS_DIR, user)
            neg_user_path = os.path.join(NEG_DIR, user)
            pos_user_samples = os.listdir(pos_user_path)
            neg_user_samples = os.listdir(neg_user_path)
            
            # merge them
            pos_user_samples, neg_user_samples = cut_samples(pos_user_samples, neg_user_samples)
            
            # copy files
            for pos_sample in pos_user_samples:
                copy_files(os.path.join(pos_user_path, pos_sample), DEST_DIR)
            for neg_sample in neg_user_samples:
                copy_files(os.path.join(neg_user_path, neg_sample), DEST_DIR)

            # remove them from the list of files to add
            pos_samples.remove(user)
            neg_samples.remove(user)
    
    # Copy the remainder of files
    print("Copying files for Pos users")
    for pos_user in pos_samples:
        print("Copying files for ", pos_user)
        user_path = os.path.join(POS_DIR, pos_user)
        user_samples = os.listdir(user_path)
        user_samples.sort()
        user_samples = user_samples[:MAX_SAMPLES_PER_USER]
        for sample in user_samples:
            copy_files(os.path.join(user_path, sample), DEST_DIR)
    print("Copying files for Neg user")
    for neg_user in neg_samples:
        print("Copying files for ", neg_user)
        user_path = os.path.join(NEG_DIR, neg_user)
        user_samples = os.listdir(user_path)
        user_samples.sort()
        user_samples = user_samples[:MAX_SAMPLES_PER_USER]
        for sample in user_samples:
            copy_files(os.path.join(user_path, sample), DEST_DIR)


if __name__ == "__main__":
    main()
