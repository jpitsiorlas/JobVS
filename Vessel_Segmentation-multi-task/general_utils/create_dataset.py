from multiprocessing import Value
import os
import copy
import json
import utils
import shutil
import numpy as np
import pandas as pd

from math import pi

from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.io import export_png, export_svg
from bokeh.models import ColumnDataSource, FactorRange

from arguments import ArgsInit

def getCount(listOfElems, cond):
    """ Count elements in a list given a condition """
    count = sum(cond(elem) for elem in listOfElems)

    return count 

def organize_freqs(percentage, ranges):
    """ Compute percentage frequencies in the given ranges """
    frequency = []
    for i, j in zip(ranges[:-1], ranges[1:]):
        count = getCount(percentage, lambda x: x >= i and x < j)
        frequency.append(count)
    count = getCount(percentage, lambda x: x >= j)
    frequency.append(count)

    return frequency

def plot_folds(fold1, fold2, partition=5):
    """ Plot the vessel segmentation percentage frequency by fold """
    # We take the vessel percentage from the fold lists
    fold1 = [i["vessel_percentage"]*100 for i in fold1]
    fold2 = [i["vessel_percentage"]*100 for i in fold2]
    
    min_p, max_p = min(min(fold1), min(fold2)), max(max(fold1), max(fold2))
    ranges = np.linspace(min_p, max_p, partition)
    freqs_f1 = organize_freqs(fold1, ranges)
    freqs_f2 = organize_freqs(fold2, ranges)
    
    freqs = list( map(round, ranges, [3]*len(ranges)))
    ranges = [str(round(i,3)) + " - " + str(round(j-0.001,3)) for i, j in zip(freqs[:-1], freqs[1:])]
    ranges.append(str(freqs[-1]))
    cross_val = ["fold_1", "fold_2"]

    data = {"ranges"   : ranges,
            "fold_1": freqs_f1,
            "fold_2"  : freqs_f2}

    color_fold1 = '#C8E7FF'
    color_fold2 = '#DEAAFF'
    
    factors = [(rng, abl) for rng in ranges for abl in cross_val]
    x = [(rng, "fold_1") for rng in ranges]
    counts = data["fold_1"]
    source = ColumnDataSource(data=dict(x=x, counts=counts))

    x1 = [(rng, "fold_2") for rng in ranges]
    counts1 = data["fold_2"]
    source1 = ColumnDataSource(data=dict(x=x1, counts=counts1))

    p = figure(x_range=FactorRange(*factors),y_range=(0,14),
               plot_height=500, plot_width=600,
               toolbar_location=None, tools="")
    
    # Plot frequencies
    p.vbar(x=dodge("x", 0, range=p.x_range),
           top="counts",width=0.9, source=source, 
           color=color_fold1, legend_label="Fold 1")
    
    p.vbar(x=dodge("x", 0, range=p.x_range),
           top="counts", width=0.9, source=source1, 
           color=color_fold2, legend_label="Fold 2")

    # legend settings
    p.legend.location = "top_right"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "16px"

    # x axis settings
    p.x_range.range_padding = 0
    p.xaxis.major_label_text_alpha = 0
    p.xaxis.major_label_text_font_size = "0px"
    p.xaxis.group_label_orientation = pi/2
    p.xaxis.group_text_font_size = "13.5px"
    p.xgrid.grid_line_color = None
    
    # y axis settings
    p.xaxis.axis_label = "Vessel percentage ranges (%)"
    p.yaxis.axis_label = "Number of MRIs in the vessel percentage range"

    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_text_font_size = '13px'
    p.yaxis.major_label_text_font_size = "13.5px"
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_text_font_size = '13px'

    p.ygrid.minor_grid_line_color = 'black'
    p.ygrid.minor_grid_line_alpha = 0.1

    # save .svg
    p.output_backend = "svg"
    
    return p, (min_p, max_p, partition)


def main(args):
    json_file = os.path.join(args.dataset_path, "data.json")
    data = utils.load_json(json_file)
    # Split random for balanced cross validation in data
    pd_data = pd.DataFrame(data["data"])
    #pd_data = pd_data.sort_values(by="vessel_percentage")
    list_data = [{"image": im, "label": lab} 
                 for im, lab
                 in zip(pd_data["image"], pd_data["label"],
                        )]
    fold1 = list_data[::2]
    fold2 = list_data[1::2]

    f1_json = copy.deepcopy(data)
    f1_json["training"] = fold1
    f1_json["validation"] = fold2
    del f1_json["data"]
    
    f2_json = copy.deepcopy(data)
    f2_json["training"] = fold2
    f2_json["validation"] = fold1
    del f2_json["data"]
    
    # Plot data distribution in the two folds
    # fig_dir = os.path.join(args.dataset_path, "data_distribution")
    # p, ranges = plot_folds(fold1=fold1, fold2=fold2)
    
    # f1_json["ranges"] = {"min": ranges[0],
    #                      "max": ranges[1],
    #                      "steps": ranges[2]
    #                     }
    # f2_json["ranges"] = f1_json["ranges"]
    # export_png(p, filename=fig_dir+'.png')
    # export_svg(p, filename=fig_dir+'.svg')

    json_dir = os.path.join(args.dataset_path, "data_fold1.json")
    utils.save_json(f1_json, json_dir)
    print('Json file saved at =====> ', json_dir)

    json_dir = os.path.join(args.dataset_path, "data_fold2.json")
    utils.save_json(f2_json, json_dir)
    print('Json file saved at =====> ', json_dir)
    

if __name__ == "__main__":
    args = ArgsInit().return_args()
    main(args)