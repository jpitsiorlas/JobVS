# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from medpy.metric import binary
from medpy.io import load
import argparse
import numpy as np


# def dice(ref, result):
#     return binary.dc(result, ref)


# def hd(ref, result, spacing, connectivity=1):
#     return binary.hd(result, ref, spacing, connectivity)


# def balanced_average_hd(ref, result, spacing=None, connectivity=1):
#     """
#     Balanced Average Hausdorff distance. Implementation based on the following paper
#     Aydin OU, et al. On the usage of average Hausdorff distance for segmentation performance assessment: hidden error when used for ranking.
#     Eur Radiol Exp. 2021 Jan 21;5(1):4. doi: 10.1186/s41747-020-00200-2. Erratum in: Eur Radiol Exp. 2022 Oct 31;6(1):56
#     :param ref: ground truth image
#     :param result: segmentation
#     :param spacing: voxel spacing
#     :param connectivity: required by the base function. Default is one
#     :return: Balanced Hausdorff distance
#     """
#     gt_to_s = binary.__surface_distances(result, ref, spacing, connectivity)
#     s_to_gt = binary.__surface_distances(ref, result, spacing, connectivity)

#     gt = np.count_nonzero(ref)
#     balanced = (np.sum(gt_to_s) + np.sum(s_to_gt)) / gt

#     return balanced


# def average_hd(ref, result, spacing=None, connectivity=1):
#     """
#     Average Hausdorff distance. Implementation based on the following paper
#     Aydin OU, et al. On the usage of average Hausdorff distance for segmentation performance assessment: hidden error when used for ranking.
#     Eur Radiol Exp. 2021 Jan 21;5(1):4. doi: 10.1186/s41747-020-00200-2. Erratum in: Eur Radiol Exp. 2022 Oct 31;6(1):56
#     :param ref: ground truth image
#     :param result: segmentation
#     :param spacing: voxel spacing
#     :param connectivity: required by the base function. Default is one
#     :return: Balanced Hausdorff distance
#     """
#     gt_to_s = binary.__surface_distances(result, ref, spacing, connectivity)
#     s_to_gt = binary.__surface_distances(ref, result, spacing, connectivity)

#     gt = np.count_nonzero(ref)
#     s = np.count_nonzero(result)
#     average = (np.sum(gt_to_s) / gt) + (np.sum(s_to_gt) / s)

#     return average


# def jaccard(ref, result):
#     return binary.jc(result, ref)


# def volumetric_similarity(ref, result):
#     """
#     Estimates the volumetric simularity based on the following publication:
#     Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
#     BMC Med Imaging. 2015 Aug 12;15:29. doi: 10.1186/s12880-015-0068-x
#     :param ref: Ground truth
#     :param result: Segmentation
#     :return: Volumetric similarity
#     """
#     gt = np.count_nonzero(ref)
#     s = np.count_nonzero(result)

#     vs = 1-np.abs(gt - s) / (gt + s)

#     return vs


# def mutual_information(ref, result):
#     """
#     Estimates the mutual information based on the following publication:
#     Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
#     BMC Med Imaging. 2015 Aug 12;15:29. doi: 10.1186/s12880-015-0068-x
#     :param ref: Ground truth
#     :param result: Segmentation
#     :return: Volumetric similarity
#     """
#     gt_vec = ref.flatten()
#     s_vec = result.flatten()

#     tp = np.count_nonzero((gt_vec != 0) & (s_vec != 0))
#     tn = np.count_nonzero((gt_vec == 0) & (s_vec == 0))
#     fp = np.count_nonzero((gt_vec == 0) & (s_vec != 0))
#     fn = np.count_nonzero((gt_vec != 0) & (s_vec == 0))
#     n = len(gt_vec)

    

#     # Estimating entropy of the ground truth
#     p_gt_1 = (tp + fn) / n
#     p_gt_2 = (tn + fn) / n
#     entropy_gt = -((p_gt_1 * np.log(p_gt_1)) + (p_gt_2 * np.log(p_gt_2)))

#     # Estimating entropy of the segmentation
#     p_s_1 = (tp + fp) / n
#     p_s_2 = (tn + fp) / n
#     entropy_s = -((p_s_1 * np.log(p_s_1)) + (p_s_2 * np.log(p_s_2)))

#     # Estimating joint entropy
#     p_g_1_s_1 = tp / n
#     p_g_1_s_2 = fn / n
#     p_g_2_s_1 = fp / n
#     p_g_2_s_2 = tn / n

#     joint_ent = -((p_g_1_s_1 * np.log(p_g_1_s_1)) + (p_g_1_s_2 * np.log(p_g_1_s_2))
#                   + (p_g_2_s_1 * np.log(p_g_2_s_1)) + (p_g_2_s_2 * np.log(p_g_2_s_2)))

#     mi = entropy_gt + entropy_s - joint_ent

#     return mi


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute the dice score.')
    parser.add_argument('-gt', dest='reference')
    parser.add_argument('-s', dest='segmentation')
    parser.add_argument('-o', dest='operation')

    args = parser.parse_args()
    reference_file = args.reference
    segmentation_file = args.segmentation
    operation = args.operation

    reference, hdr = load(reference_file)
    segmentation, _ = load(segmentation_file)

    if operation == 'all':
        print('dice',dice(reference.flatten(), segmentation.flatten()))
        print('jaccard',jaccard(reference.flatten(), segmentation.flatten()))
        print('average_hd', average_hd(reference, segmentation, hdr.get_voxel_spacing()))
        print('balanced_average_hd', balanced_average_hd(reference, segmentation, hdr.get_voxel_spacing()))
        print('volumetric_similarity',volumetric_similarity(reference, segmentation))
        print('mutual_information',mutual_information(reference, segmentation))
    elif operation == 'dice':
        print(dice(reference.flatten(), segmentation.flatten()))
    elif operation == 'jac':
        print(jaccard(reference.flatten(), segmentation.flatten()))
    elif operation == 'hd':
        print(balanced_average_hd(reference, segmentation, hdr.get_voxel_spacing()))
    elif operation == 'vs':
        print(volumetric_similarity(reference, segmentation))
    elif operation == 'mi':
        print(mutual_information(reference, segmentation))
    else:
        print('Error: operation not defined')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
