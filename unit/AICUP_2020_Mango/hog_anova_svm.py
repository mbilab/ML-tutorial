import os
import numpy as np
import re
import cv2
from glob import glob
from skimage.feature import hog
from skimage import data, exposure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


label2idx = {
    '不良-乳汁吸附': 0,
    '不良-機械傷害': 1,
    '不良-炭疽病': 2,
    '不良-著色不佳': 3,
    '不良-黑斑病': 4
}

def load_mango_csv(csv_path='./C2_TrainDev/train.csv'):
    path = []
    box = []
    label = []
    subdir = csv_path.split('/')[-1].split('.')[0].capitalize()
    with open(csv_path, 'r', encoding='utf8') as f:        
        for line in f:
            clean_line = re.sub(',+\n', '', line).replace('\n', '').replace('\ufeff', '').split(',')
            curr_img_path = f'./C2_TrainDev_Toy/{subdir}/{clean_line[0]}'
            curr_info = np.array(clean_line[1:]).reshape(-1, 5)
            curr_box = curr_info[:, :-1].astype('float16').tolist()
            curr_label = curr_info[:, -1].tolist()
            path.append(curr_img_path)
            box.append(curr_box)
            label.append(curr_label)
    return path, box, label


def get_hog_feature(img_path, resize_shape=(224, 224), slice_img=False):
    '''
    Args:
        img_path: Load image from this path.
        resize_shape: See function get_category_data for detail description.
        slice_img: If True, divide origin images into 3*4 parts, each part of image will have its own HOG feature extracted, we then concatenate 12 parts of features into one complete feature vector.

    '''

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize_shape)

    if slice_img:
        y_part = img.shape[0]//3
        x_part = img.shape[1]//4
        hog_feat = []
        for y_idx in range(3):
            for x_idx in range(4):
                curr_img = img[y_idx*y_part:(y_idx+1)*y_part, x_idx*x_part:(x_idx+1)*x_part]
                curr_hog = hog(curr_img, block_norm='L2-Hys', pixels_per_cell=(8, 8), transform_sqrt=True, multichannel=True)
                hog_feat.append(curr_hog)
        hog_feat = np.array(hog_feat).reshape(-1)
    else:
        hog_feat = hog(img, block_norm='L2-Hys', pixels_per_cell=(16, 16), transform_sqrt=True, multichannel=True)
    return hog_feat


def get_category_data(path, box, label, cat_type, drop_non_defective=False, resize_shape=(128, 72), slice_img=False):
    '''
    Args:
        path: List of image paths.
        box: List of boxes.
        label: List of labels.
        cat_type: Defective category, same as parameter defective_type in function HOG_ANOVA_SVM.
        drop_non_defective: If True, we remove those non-defective samples due to sparse number of defective samples. This parameter should be False when building development data.
        resize_shape: Each image must have the same shape, also we shrink the image to reduce the size of extracted features.
        slice_img: See function get_hog_feature for detail description.

    '''

    assert cat_type in label2idx
    x = []
    y = []
    for i in range(len(path)):
        if i%100 == 0:
            print(i)

        if drop_non_defective and not len(label[i]):
            continue

        # use TRY because some images are missing in the image folder
        try:
            curr_feat = get_hog_feature(path[i], resize_shape=resize_shape, slice_img=slice_img)
            curr_label = 1 if cat_type in set(label[i]) else 0
            x.append(curr_feat)
            y.append(curr_label)
        except:
            continue

    return x, y


def HOG_ANOVA_SVM(file_prefix, defective_type, anova_percentile=50, slice_img=False, linear_svm=False):
    '''
    Description:
        Build a 

    Args:
        file_prefix: Any file name you want.
        defective_type: One of the five defective, Chinese required.
        anova_percentile: Percent of HOG features you select to feed into SVM.
        slice_img: See function get_hog_feature for detail description.
        linear_svm: If True, use LinearSVC instead of SVC with rbf kernel.

    Examples:
        >>> defective_type = '不良-乳汁吸附'
        >>> file_prefix = f'{defective_type.split('-')[-1]}'
        >>> HOG_ANOVA_SVM(file_prefix, defective_type, anova_percentile=5, slice_img=True, linear_svm=False)

    '''

    assert defective_type in label2idx

    print('build train data...')
    train_file_name = f'./{file_prefix}_train'
    train_path, train_box, train_label = load_mango_csv(csv_path='./C2_TrainDev_Toy/train.csv')
    if os.path.isfile(f'{train_file_name}_x.npy') and os.path.isfile(f'{train_file_name}_y.npy'):
        train_x = np.load(f'{train_file_name}_x.npy')
        train_y = np.load(f'{train_file_name}_y.npy')
    else:
        train_x, train_y = get_category_data(train_path, train_box, train_label, defective_type, drop_non_defective=True, resize_shape=(128, 72), slice_img=slice_img)
        np.save(f'{train_file_name}_x', np.array(train_x))
        np.save(f'{train_file_name}_y', np.array(train_y))

    print('build dev data...')
    dev_file_name = f'./{file_prefix}_dev'
    dev_path, dev_box, dev_label = load_mango_csv(csv_path='./C2_TrainDev_Toy/dev.csv')
    if os.path.isfile(f'{dev_file_name}_x.npy') and os.path.isfile(f'{dev_file_name}_y.npy'):
        dev_x = np.load(f'{dev_file_name}_x.npy')
        dev_y = np.load(f'{dev_file_name}_y.npy')
    else:
        dev_x, dev_y = get_category_data(dev_path, dev_box, dev_label, defective_type, drop_non_defective=False, resize_shape=(128, 72), slice_img=slice_img)
        np.save(f'{dev_file_name}_x', np.array(dev_x))
        np.save(f'{dev_file_name}_y', np.array(dev_y))

    print('train linear svm model...')
    svm = LinearSVC(random_state=42, C=1.0, class_weight='balanced') if linear_svm else SVC(kernel='rbf', gamma='auto', random_state=42, C=1.0, class_weight='balanced')
    clf = Pipeline([('anova', SelectPercentile(chi2)),
                    ('scaler', StandardScaler()),
                    ('svc', svm)])

    clf.set_params(anova__percentile=anova_percentile)
    clf.fit(train_x, train_y)
    dev_pred_y = clf.predict(dev_x)

    print('evaluating dev data...')
    cm = confusion_matrix(dev_y, dev_pred_y)
    acc = accuracy_score(dev_y, dev_pred_y)
    f1 = f1_score(dev_y, dev_pred_y)
    p = precision_score(dev_y, dev_pred_y)
    r = recall_score(dev_y, dev_pred_y)

    print(cm, acc, f1, p, r)


if __name__ == '__main__':
    # defective_type = '不良-乳汁吸附'
    # file_prefix = f'{defective_type.split('-')[-1]}'
    # HOG_ANOVA_SVM(file_prefix, defective_type, anova_percentile=5, slice_img=True, linear_svm=False)
    exit()