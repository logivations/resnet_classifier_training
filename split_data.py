import os
import shutil

src_folder = "/data/tlt_training/faurecia/data/new/all_crops"
target_path = "/data/tlt_training/faurecia/data/new/all_crops_split"
for subdir in ("empty_slot", "full_slot"):
    for i in range(1, 14):
        train_dir = os.path.join(target_path, "train", subdir, str(i))
        val_dir = os.path.join(target_path, "val", subdir, str(i))
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        im_dir = os.path.join(src_folder, subdir, str(i))
        im_names = os.listdir(im_dir)
        n_validation = min(len(im_names) // 10, 6)
        validation_names = im_names[-n_validation:]
        train_names = im_names[:-n_validation]
        for name in train_names:
            shutil.copy(os.path.join(im_dir, name), os.path.join(train_dir, name))
        for name in validation_names:
            shutil.copy(os.path.join(im_dir, name), os.path.join(val_dir, name))



