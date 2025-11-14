#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

from utils.image_utils import (_load_maybe_dict_npy, _local_to_global_mapping,
                               get_rgb_image, _obtain_spectral_resolution,
                               adjust_spectral_data, _compute_attenuation,
                               _compute_transmittance, _load_hadar_emissivity
                               )
from utils.blackbody import blackbody


def generate_heatcubes(root_dir, out_root=None, scene_filter=None):
    """
    Processes each 1080×1920 sample as a whole without patch splitting.
    Runs forward model and saves the resulting heatcube along with the GroundTruth files.
    GroundTruth and simulated HeatCubes are saved under /data/abdar/<scene name>/.
    
    Params:
    root_dir: Directory containing the raw HADAR dataset (scene folders).
    out_root: Optional output parent directory. If None the default is a sibling folder
              next to root_dir: os.path.join(parent_of_root, "abdar")
    scene_filter: Optional scene name to process only that scene (e.g., 'Scene5_Highway').
    """
    root_dir = os.path.abspath(root_dir)
    if out_root is None:
        data_dir = os.path.join(os.path.dirname(root_dir), "abdar")
    else:
        data_dir = os.path.abspath(out_root)
    os.makedirs(data_dir, exist_ok=True)

    # This valules of lambda are for the emissivity data and the definition
    # comes from the HADAR code. 49 values because that is the number of bands
    # on the synthetic data. See https://github.com/FanglinBao/HADAR
    lambda_vals_desired = np.flip(1e4 / np.linspace(720, 1250, 49, dtype=np.float32))
    target_resolution_nm = _obtain_spectral_resolution(lambda_vals_desired)

    # In this file the first column are the wavelenghts values and the second 
    # one, transmittace values
    lambda_vals_file = np.load('data/local_npys/transmittance_atten_1mAir.npy')[:, 0].astype(np.float32)
    transmittance = np.load('data/local_npys/transmittance_atten_1mAir.npy')[:, 1].astype(np.float32)

    attenuation_dict = _compute_attenuation(transmittance, 'dB_per_m')
    attenuation_from_transmittance = attenuation_dict['attenuation']
    attenuation_units = attenuation_dict['attenuation_units']


    adjusted_attenuation, lambda_vals_matched = adjust_spectral_data(np.concatenate([lambda_vals_file[:, None], attenuation_from_transmittance[:, None]], axis=1), 
                                                                       lambda_vals_desired,
                                                                       target_resolution_nm)

    B_air = blackbody(lambda_vals_matched, 293.15, 'microflicks') # ~20C

    # Process each scene folder independently
    for scene in sorted(os.listdir(root_dir)):
        if scene_filter and scene != scene_filter:
            continue  # Skip if filtering to a specific scene
        scene_input_dir = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_input_dir):
            continue

        # Create output directories for the scene
        scene_dir = os.path.join(data_dir, scene)
        gt_dir = os.path.join(scene_dir, "GroundTruth")
        hc_dir = os.path.join(scene_dir, "HeatCubes")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(hc_dir, exist_ok=True)
        tmap_dir = os.path.join(gt_dir, "tMap")
        emap_dir = os.path.join(gt_dir, "eMap")
        mapped_emap_dir = os.path.join(gt_dir, "mapped_eMap")
        depth_dir = os.path.join(gt_dir, "Depth")
        os.makedirs(tmap_dir, exist_ok=True)
        os.makedirs(emap_dir, exist_ok=True)
        os.makedirs(mapped_emap_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        # Save eList once per scene if available
        elist_src = os.path.join(scene_input_dir, "GroundTruth", "eMap", "eList.npy")
        if os.path.exists(elist_src):
            elist = np.load(elist_src, allow_pickle=True)
            np.save(os.path.join(emap_dir, "eList.npy"), elist)

        # Process samples for each scene (using 5 sample IDs and two sides)
        for i in range(1, 6):
            for side in ['L', 'R']:
                sample_id = f"{i:04d}"
                heatcube_path = os.path.join(scene_input_dir, "HeatCubes", f"{side}_{sample_id}_heatcube.npy")
                tmap_path = os.path.join(scene_input_dir, "GroundTruth", "tMap", f"tMap_{side}_{sample_id}.npy")
                emap_path = os.path.join(scene_input_dir, "GroundTruth", "eMap", f"eMap_{side}_{sample_id}.npy")
                depth_path = os.path.join(scene_input_dir, "GroundTruth", "Depth", f"Depth_{side}_{sample_id}.npy")


                # Load raw arrays using dictionary extraction
                tmap = _load_maybe_dict_npy(tmap_path, "tMap", dtype=np.uint8) # type: ignore
                emap = _load_maybe_dict_npy(emap_path, "eMap", dtype=np.uint8)  # (H,W) local 1-based # type: ignore
                depth = _load_maybe_dict_npy(depth_path, "depth", dtype=np.float32) # type: ignore
  
                mapped_emap = emap.astype(int)


                # I saved the mapped emaps as images because is cheaper in terms 
                # of storage. And when we load them we just have to take one 
                # channel. The mapped indexes remain unchanged.
                img_mapped_emap = Image.fromarray(np.expand_dims(mapped_emap, axis=2).repeat(3, axis=2).astype('uint8'))
                img_mapped_emap.save(os.path.join(mapped_emap_dir, f"mapped_eMap_{side}_{sample_id}.png"))

                # Prepare groundtruth saving with original stereo naming
                np.save(os.path.join(tmap_dir, f"tMap_{side}_{sample_id}.npy"), tmap)
                np.save(os.path.join(emap_dir, f"eMap_{side}_{sample_id}.npy"), emap)
                np.save(os.path.join(depth_dir, f"Depth_{side}_{sample_id}.npy"), depth)

                # Build inputs for forward model
                T_obj = tmap[..., None] + 273.15  # (H,W,1) in Kelvin
                d_map = depth[..., None]          # (H,W,1)

                scene_transmittance = _compute_transmittance(adjusted_attenuation, d_map,
                                                             attenuation_units)
                
                B_obj = blackbody(lambda_vals_matched, T_obj, units='microflicks')

                scene_emissivity = _load_hadar_emissivity(img_mapped_emap)
                print(f"min and max of emissivity: {np.min(scene_emissivity)}, {np.max(scene_emissivity)}")

                hyperspectral_simulated_image = scene_transmittance * \
                    (scene_emissivity * B_obj) + (1.0 - scene_transmittance) * B_air

                # Save precomputed images
                img_gray = Image.fromarray(get_rgb_image(hyperspectral_simulated_image, 'sum'))
                gray_filename = f"{side}_{sample_id}_heatcube_sim_gray.png"
                img_gray.save(os.path.join(hc_dir, gray_filename))

                img_pca = Image.fromarray(get_rgb_image(hyperspectral_simulated_image, 'pca'))
                pca_filename = f"{side}_{sample_id}_heatcube_sim_pca.png"
                img_pca.save(os.path.join(hc_dir, pca_filename))

                # Save simulated heatcube with new naming structure
                new_filename_sim = f"{side}_{sample_id}_heatcube_sim.npy"
                np.save(os.path.join(hc_dir, new_filename_sim), 
                        hyperspectral_simulated_image.astype(np.float32))
                        
    print("Synthetic heatcubes generated with scene-specific structure.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic HeatCubes from HADAR raw data.")
    parser.add_argument("--root_dir", help="Path to raw HADAR dataset root (scene folders).")
    parser.add_argument("--out_dir", help="Optional output parent directory. Defaults to sibling 'abdar' next to root_dir.")
    parser.add_argument("--scene", help="Optional: Process only this scene (e.g., 'Scene5_Highway'). If not provided, processes all scenes.")
    args = parser.parse_args()
    generate_heatcubes(args.root_dir, out_root=args.out_dir, scene_filter=args.scene)