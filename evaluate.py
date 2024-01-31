import os
import time
import yaml

import torch
import pandas as pd
from torch.utils.data import DataLoader

import utils
from model.model import ColorModel
from data.dataloader import StaticDataset
from metrics.metric import PointCloudMetric

# Paths
base_path = "./results"
data_path = "./data/datasets/full_128" 

experiments = [
    #"MeanScale_4_lambda800-6400",
    "MeanScale_5_lambda200-6400_200epochs",
    "MeanScale_1_lambda300",
    "MeanScale_1_lambda600",
    "MeanScale_1_lambda800",
    "MeanScale_1_lambda1200",
]

def run_testset(experiments):
    # Device
    device = torch.device(0)
    torch.cuda.set_device(device)

    torch.no_grad()
        
    # Dataloader
    test_set = StaticDataset(data_path, split="test", transform=None, partition=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for experiment in experiments:
        weight_path = os.path.join(base_path, experiment, "weights.pt")
        config_path = os.path.join(base_path, experiment, "config.yaml")

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Model
        model = ColorModel(config["model"])
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        model.update()
        
        experiment_results = []

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                # Prepare data
                points = data["src"]["points"].to(device, dtype=torch.float)
                colors = data["src"]["colors"].to(device, dtype=torch.float)
                source = torch.concat([points, colors], dim=2)[0]
                coordinates = source.clone()

                # Side info
                N = source.shape[0]
                sequence = data["cubes"][0]["sequence"][0]
                print(sequence)

                # Compression
                torch.cuda.synchronize()
                t0 = time.time()
                strings, shapes = model.compress(source)
                torch.cuda.synchronize()
                t_compress = time.time() - t0

                # Decompress all rates
                y_strings = []
                z_strings = []
                for i in range(len(strings[0])):
                    y_strings.append(strings[0][i])
                    z_strings.append(strings[1][i])
                    current_strings = [y_strings, z_strings]

                    # Run decompression
                    torch.cuda.synchronize()
                    t0 = time.time()
                    reconstruction = model.decompress(coordinates=coordinates, 
                                                            strings=current_strings, 
                                                            shape=shapes)
                    torch.cuda.synchronize()
                    t_decompress = time.time() - t0
                    
                    # Rebuild point clouds
                    source_pc = utils.get_o3d_pointcloud(source)
                    rec_pc = utils.get_o3d_pointcloud(reconstruction)

                    # Compute metrics
                    metric = PointCloudMetric(source_pc, rec_pc)
                    results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=True)

                    # Save results
                    results["bpp"] = utils.count_bits(y_strings) / N
                    results["layer"] = i
                    results["sequence"] = data["cubes"][0]["sequence"][0]
                    results["frameIdx"] = data["cubes"][0]["frameIdx"][0].item()
                    results["t_compress"] = t_compress
                    results["t_decompress"] = t_decompress
                    experiment_results.append(results)

                    # Renders
                    path = os.path.join(base_path,
                                        experiment, 
                                        "renders_test", 
                                        "{}_{}_{}.png".format(sequence, str(i), "{}"))
                    utils.render_pointcloud(rec_pc, path)

                    # Ply
                    path = os.path.join(base_path,
                                        experiment, 
                                        "plys", 
                                        "{}_{:04d}_rec{}.ply".format(sequence, results["frameIdx"], str(i)))

        # Save the results as .csv
        df = pd.DataFrame(experiment_results)
        results_path = os.path.join(base_path, experiment, "test.csv")
        df.to_csv(results_path)

if __name__ == "__main__":
    run_testset(experiments)