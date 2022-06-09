import os
import shutil

src_folder = "/data/FST/demo/dataset/all"
target_path = "/data/FST/demo/dataset/split"
for subdir in ("box", "nobox"):
    for i in "blue", "gray":
        train_dir = os.path.join(target_path, "train", subdir, str(i))
        val_dir = os.path.join(target_path, "val", subdir, str(i))
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        im_dir = os.path.join(src_folder, subdir, str(i))
        im_names = os.listdir(im_dir)
        n_validation = max(len(im_names) // 10, 6)
        validation_names = im_names[-n_validation:]
        train_names = im_names[:-n_validation]
        for name in train_names:
            shutil.copy(os.path.join(im_dir, name), os.path.join(train_dir, name))
        for name in validation_names:
            shutil.copy(os.path.join(im_dir, name), os.path.join(val_dir, name))



