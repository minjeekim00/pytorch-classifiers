import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os, shutil
from sklearn.model_selection import train_test_split

def find_classes(dir):
    """
       returns classes, class_to_idx
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_idxs_from_data(dir, classes):
    """
        dir = image directory
        classes = list of class
        getting idx from all class directory
        item name should be (idx)_*
    """
    idxs = []
    
    if type(classes) is str:
        class_dir = os.path.join(dir, classes)
        
        v_name_dirs = os.listdir(class_dir)
        for vname in v_name_dirs:
            if vname in classes:
                continue
            idx = vname.split('_')[0]
            if idx not in idxs:
                idxs.append(idx)
        return idxs
    
    elif type(classes) is list:
        for c in classes:
            class_dir = os.path.join(dir, c)

            v_name_dirs = os.listdir(class_dir)
            for vname in v_name_dirs:
                if vname in classes:
                    continue
                idx = vname.split('_')[0]
                if idx not in idxs:
                    idxs.append(idx)
        return idxs
    else :
        print("input type is not list nor string")
        return

def divide_by_categories(dir, df):
    print("moving to class directories....")
    for row in df.values:
        idx = row[0]
        label = row[1]
        
        dst_dir = os.path.join(dir, label)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        
        source = os.listdir(dir)
        for files in source:
            if files.startswith(str(idx)+"_") and files.endswith(".jpg"):
                shutil.move(os.path.join(dir, files), dst_dir)
    print("moving done. classes: {}".format(find_classes(dir)[0]))
    return

def split_train_val_test(dir, df, val_size, test_size):
    """
        str_col_id = column name of index data
        str_col_label = column name of label data
    """
    divide_by_categories(dir, df)
    classes = find_classes(dir)[0]
    
    for c in classes:
        class_dir = os.path.join(dir, c)
        
        if not os.path.exists(class_dir):
            print("{} not exist".format(class_dir))
            continue
        
        idxs = get_idxs_from_data(dir, c)
        print("class: {}".format(c))
        x_train, x_test = train_test_split(idxs, test_size=test_size)
        x_train, x_val = train_test_split(x_train, test_size=val_size)
        
        for phase in ["train", "val", "test"]:
            target_dir = os.path.join(dir, phase)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            if phase == "train":
                target = x_train
            elif phase == "val":
                target = x_val
            elif phase == "test":
                target = x_test

            for vid in tqdm(target):
                origin_dir = os.path.join(class_dir, str(vid)+'_*')
                dst_dir = os.path.join(target_dir, os.path.basename(class_dir))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                
                os.system("mv %s %s" % (origin_dir, dst_dir))
            
        os.rmdir(class_dir)
    return
