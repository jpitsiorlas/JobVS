import os
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage.morphology import binary_erosion
from scipy.spatial.distance import cdist
from libs.dataloader import helpers
from libs.utilities.utils import draw_curve
from skimage.morphology import skeletonize,skeletonize_3d

from sklearn.metrics import average_precision_score, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def test(folder, root_dir, csv_file, classes, detection=False):
    # ! Modified to read annotations in UNETR format
    # ! May 25th by N. Valderrama.
    
    data = helpers.convert_format(csv_file, val=True)
    labels = data['label'].tolist()
    images = data['image'].tolist()
    images = [img.replace('imagesTs/', '') for img in images]
    images.sort()
    labels.sort()

    if detection:
        patients = Parallel(n_jobs=10)(
            delayed(parallel_test_det)(images[j], labels[j], folder, root_dir, classes)
            for j in range(len(labels)))    
        name_classes = ['Vessels', 'Vessels_with_Brain']
        fields = ['Label'] + ['mAP_' + i for i in name_classes]
        fields = fields + ['Recall_' + i for i in name_classes]
        fields = fields + ['Precision_' + i for i in name_classes]
        mean = np.zeros([len(patients), (classes - 1) * 3])
        curves = [patient[7:] for patient in patients]
        patients = [patient[:7] for patient in patients]
        maps = [patient[1:] for patient in patients]
        maps= np.asarray(maps)
        maps = [np.mean(maps, axis=0)[0], np.mean(maps, axis=0)[1]]
        curves = np.asarray(curves)
        curves = np.nan_to_num(np.mean(curves, axis=0))
        draw_curve(scores=maps, metrics=curves, res=name_classes, save_path=folder)
        save_name = '/mean_AP'
    else:
        patients = Parallel(n_jobs=10)(
            delayed(parallel_test)(images[j], labels[j], folder, root_dir, classes)
            for j in range(len(labels)))
        
        name_classes = ['Brain', 'Vessels', 'Vessels_with_Brain']
        fields = ['Label'] + ['mAP_' + i for i in name_classes]
        fields = fields + ['Recall_' + i for i in name_classes]
        fields = fields + ['Precision_' + i for i in name_classes]
        mean = np.zeros([len(patients), (classes - 1) * 3])
        save_name = '/dice'

    with open(folder + save_name + '.csv', 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fields)
        writer.writeheader()

        for idx, j in enumerate(patients):
            line = {field: datum for field, datum in zip(fields, j)}
            mean[idx] = np.asarray(j[1:])
            writer.writerow(line)

        mean = np.nanmean(mean, axis=0)
        last = ['mean'] + list(mean)
        writer.writerow({field: datum for field, datum in zip(fields, last)})
        outcsv.close()
    print(mean)


def read_image(path):
    im = nib.load(path)
    affine = im.affine
    im = im.get_fdata()
    return im, affine


def dice_score(im, lb):
    lb_f = np.ndarray.flatten(lb)
    im_f = np.ndarray.flatten(im)

    tps = np.sum(im * lb)
    fps = np.sum(im * (1 - lb))
    fns = np.sum((1 - im) * lb)
    labels = np.sum(lb_f)
    pred = np.sum(im_f)

    if labels == 0 and pred == 0:
        dice = 1
    else:
        dice = (2 * tps) / (2 * tps + fps + fns)
    rec = tps / (tps + fns)
    prec = tps / (tps + fps)
    return dice, rec, prec


def find_border(data):
    eroded = binary_erosion(data)
    border = np.logical_and(data, np.logical_not(eroded))
    return border


def get_coordinates(data, affine):
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    indices = np.vstack(np.nonzero(data))
    indices = np.vstack((indices, np.ones(indices.shape[1])))
    coordinates = np.dot(affine, indices)
    return coordinates[:3, :]


def eucl_max(nii1, nii2, affine):
    origdata1 = np.logical_not(np.logical_or(nii1 == 0, np.isnan(nii1)))
    origdata2 = np.logical_not(np.logical_or(nii2 == 0, np.isnan(nii2)))

    if origdata1.max() == 0 or origdata2.max() == 0:
        return np.NaN

    border1 = find_border(origdata1)
    border2 = find_border(origdata2)

    set1_coordinates = get_coordinates(border1, affine)
    set2_coordinates = get_coordinates(border2, affine)
    distances = cdist(set1_coordinates.T, set2_coordinates.T)
    mins = np.concatenate((np.amin(distances, axis=0),
                           np.amin(distances, axis=1)))
    return np.percentile(mins, 95)


def task(im, classes):
    tasks = []
    for i in range(classes):
        temp = im == i
        tasks.append(temp)
    return tasks


def parallel_test(im, lb, test_dir, root_dir, classes):
    dice = []
    precision = []
    recall = []
    clDs = []
    # hausdorff = []
    print(im)
    name = im[:-4]
    
    # Brain
    im_path = os.path.join(test_dir, 'brain', im)
    lb_path = os.path.join(root_dir.replace(
                            'Vessel_Segmentation/original', 'Brain_Segmentation/original'),
                            lb)

    image, affine = read_image(im_path)
    label, _ = read_image(lb_path)
    label = np.round(label)

    im_task = task(image, classes)[1]
    lb_task = task(label, classes)[1]

    dc, rec, prec = dice_score(im_task, lb_task)
    dice.append(dc)
    precision.append(prec)
    recall.append(rec)
    
    # Vessel
    im_path = os.path.join(test_dir, 'vessels', im)
    lb_path = os.path.join(root_dir, lb)

    image, affine = read_image(im_path)
    label, _ = read_image(lb_path)
    label = np.round(label)

    im_task = task(image, classes)[1]
    lb_task = task(label, classes)[1]

    dc, rec, prec = dice_score(im_task, lb_task)
    dice.append(dc)
    precision.append(prec)
    recall.append(rec)
    cl = clDice(image,label)
    clDs.append(cl)
    
    # Vessel
    im_path = os.path.join(test_dir, 'vessels_brain', im)
    lb_path = os.path.join(root_dir, lb)

    image, affine = read_image(im_path)
    label, _ = read_image(lb_path)
    label = np.round(label)

    im_task = task(image, classes)[1]
    lb_task = task(label, classes)[1]

    dc, rec, prec = dice_score(im_task, lb_task)
    dice.append(dc)
    precision.append(prec)
    recall.append(rec)
    
    return [name] + dice + recall + precision + clDs


def parallel_test_det(im, lb, test_dir, root_dir, classes):
    all_prec = []
    all_rec = []
    precision = []
    recall = []
    mean_AP = []
    # hausdorff = []
    print(im)
    name = im[:-4]
    
    
    # Vessel
    
    for exp in ['vessels_logits', 'vessels_brain_logits']:
        Precision = []
        Recall = []
        dice = []
        im_path = os.path.join(test_dir, exp, im)
        lb_path = os.path.join(root_dir,lb) 
        # lb_path = os.path.join(root_dir.replace(
        #                     'original', 'mask'), lb)

        image, affine = read_image(im_path)
        label, _ = read_image(lb_path)
        label = np.round(label)

        y_true = label.ravel()
        scores = image.ravel()
        ap = average_precision_score(y_true, scores)
        mean_AP.append(ap)
        thresholds = np.linspace(0, 1, 51)
        for i in thresholds:
            im_thr = image >= i
            dc, rec, prec = dice_score(im_thr, label)
            dice.append(dc)
            Recall.append(rec)
            Precision.append(prec)
            
        all_prec.append(Precision)
        all_rec.append(Recall)
        precision.append(Precision[np.argmax(dice)])
        recall.append(Recall[np.argmax(dice)])
    # max_thr = thresholds[np.argmax(dice)]
    
    return [name] + mean_AP + recall + precision + all_rec + all_prec 
    # return [name, dice, recall, precision, ap, max_thr]