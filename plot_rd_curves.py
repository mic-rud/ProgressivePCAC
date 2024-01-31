import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib.pyplot as plt
import pandas as pd

path = "./results"
sequences = ["loot", "redandblack", "longdress", "soldier"]
data_points = {
    #"Baseline" : ["Scales_1_lambda400", "Scales_1_lambda800", "Scales_1_lambda1600", "Scales_1_lambda3200", ],
    "Mean-Scale-Hyperprior" : ["MeanScale_1_lambda200", "MeanScale_1_lambda300", "MeanScale_1_lambda400", "MeanScale_1_lambda600", "MeanScale_1_lambda800", "MeanScale_1_lambda1200", "MeanScale_1_lambda1600",],# "Scales_1_lambda3200", ],
    #"long" : "Scales_5_lambda20-12500",
    #"extra" : "Scales_5_lambda200-3200_extra",
    #"logperBN" : "Scales_5_lambda200-3200_logPerBN",
    #"wide" : "Scales_5_lambda200-3200_wide",
    #"equal" : "Scales_5_lambda200-3200_wide_equalLoss",
    #"R_once" : "Scales_5_lambda200-3200_wide_equalLoss_rate_once",
    #"SUMMED" : "Scales_5_lambda200-3200_summed_log",
    #"ScaledTest" : "Scaled_test",
    #"Progressive (100)" : "MeanScale_5_lambda200-6400",
    #"Progressive (Ours)" : "MeanScale_5_lambda800-6400",
    #"Progressive-4(Ours)" : "MeanScale_4_lambda800-6400",
    "Progressive - 200(Ours)" : "MeanScale_5_lambda200-6400_200epochs",
}

cumsum = {
    "Mean-Scale-Hyperprior" : ["MeanScale_1_lambda200", "MeanScale_1_lambda400", "MeanScale_1_lambda800", "MeanScale_1_lambda1600"]
          }
BD_reference = ["Mean-Scale-Hyperprior"]

# Load data
data_frames = {}

for key, data_point in data_points.items():
    data = pd.DataFrame()
    if isinstance(data_point, list):
        for d in data_point:
            data_path = os.path.join(path, d, "test.csv")
            new_data = pd.read_csv(data_path)
            new_data["name"] = d
            data = pd.concat([data, new_data])
        
    else:
        data_path = os.path.join(path, data_point, "test.csv")
        data = pd.read_csv(data_path)

    data_frames[key] = data

for sequence in sequences:
    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]
        
        ax.plot(data["bpp"], data["sym_y_psnr"], 
                 label=key,
                 marker="x" )
        
        #bd_model = Bjontegaard_Model(data["bpp"], data["sym_y_psnr"])
        #x_scat, y_scat, x_dat, y_dat = bd_model.get_plot_data()
        #plt.scatter(x_scat, y_scat) #color=colors[i], marker=markers[i])
        #plt.plot(x_dat, y_dat) #color=colors[i], linestyle=linestyles[i], label=key)
                

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("Y PNSR [dB]")
    plt.tight_layout()
    fig.savefig("plot/figures/{}_Y-PSNR.pdf".format(str(sequence)))


    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]
        
        ax.plot(data["bpp"], (data["sym_y_psnr"] * 6 + data["sym_u_psnr"] + data["sym_v_psnr"]) / 8, 
                 label=key,
                 marker="x" )
        
        #bd_model = Bjontegaard_Model(data["bpp"], data["sym_y_psnr"])
        #x_scat, y_scat, x_dat, y_dat = bd_model.get_plot_data()
        #plt.scatter(x_scat, y_scat) #color=colors[i], marker=markers[i])
        #plt.plot(x_dat, y_dat) #color=colors[i], linestyle=linestyles[i], label=key)
                

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("YUV PNSR [dB]")
    plt.tight_layout()
    fig.savefig("plot/figures/{}_YUV-PSNR.pdf".format(str(sequence)))


    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]
        
        if key in cumsum.keys():
            run = cumsum[key]
            data = data[data["name"].isin(run)]
            rates = data["bpp"].cumsum()
        else:
            rates = data["bpp"]
        ax.plot(rates, data["sym_y_psnr"], 
                 label=key,
                 marker="x" )
        
        #bd_model = Bjontegaard_Model(data["bpp"], data["sym_y_psnr"])
        #x_scat, y_scat, x_dat, y_dat = bd_model.get_plot_data()
        #plt.scatter(x_scat, y_scat) #color=colors[i], marker=markers[i])
        #plt.plot(x_dat, y_dat) #color=colors[i], linestyle=linestyles[i], label=key)
                

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("Y PNSR [dB]")
    plt.tight_layout()
    fig.savefig("plot/figures/{}_Y-PSNR_cumsum.pdf".format(str(sequence)))

    continue
    ### Bjontegaard Models
    reference_data = data_frame[BD_reference]
    reference_data_cumulative = data_frame[BD_reference]
    reference_data_cumulative["bpp"] = reference_data_cumulative["bpp"].cumsum

    reference_model = Bjontegaard_Model(reference_data["bpp"], reference_data["sym_y_psnr"])
    reference_model_cumulative = Bjontegaard_Model(reference_data["bpp"], reference_data["sym_y_psnr"])
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]