import argparse
from pathlib import Path

import numpy as np

from forward_models.blackbody import W_m2_sr_cm1_to_microflicks
from utils.image_utils import _load_maybe_dict_npy, _local_to_global_mapping


# Wavenumbers (cm^-1) and corresponding wavelengths (µm) used in HADAR
WAVENUMBERS_CM1 = np.linspace(720.0, 1250.0, 49, dtype=np.float64)
WAVELENGTHS_UM = 1e4 / WAVENUMBERS_CM1


def process_scene(scene_dir: Path, out_dir: Path) -> None:
    """Process one scene folder: copy GT and convert heatcubes to microflicks."""
    gt_dir = out_dir / "GroundTruth"
    hc_dir = out_dir / "HeatCubes"

    tmap_dir = gt_dir / "tMap"
    emap_dir = gt_dir / "eMap"
    mapped_emap_dir = gt_dir / "mapped_eMap"
    depth_dir = gt_dir / "Depth"

    for d in (gt_dir, hc_dir, tmap_dir, emap_dir, mapped_emap_dir, depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    # eList + mapping
    elist_src = scene_dir / "GroundTruth" / "eMap" / "eList.npy"
    local_to_global = None

    if elist_src.exists():
        elist = np.load(elist_src, allow_pickle=True)
        np.save(emap_dir / "eList.npy", elist)
        local_to_global = _local_to_global_mapping(elist_src)
    else:
        print(f"[WARN] {scene_dir.name}: eList.npy not found, using local emissivity indices.")

    # 5 samples × 2 sides per scene
    for i in range(1, 6):
        for side in ("L", "R"):
            sample_id = f"{i:04d}"

            heatcube_path = scene_dir / "HeatCubes" / f"{side}_{sample_id}_heatcube.npy"
            tmap_path = scene_dir / "GroundTruth" / "tMap" / f"tMap_{side}_{sample_id}.npy"
            emap_path = scene_dir / "GroundTruth" / "eMap" / f"eMap_{side}_{sample_id}.npy"
            depth_path = scene_dir / "GroundTruth" / "Depth" / f"Depth_{side}_{sample_id}.npy"

            if not (heatcube_path.exists() and tmap_path.exists()
                    and emap_path.exists() and depth_path.exists()):
                continue

            # Load arrays (dict or raw npy)
            heatcube = _load_maybe_dict_npy(heatcube_path, "S")
            tmap = _load_maybe_dict_npy(tmap_path, "tMap")
            emap = _load_maybe_dict_npy(emap_path, "eMap")
            depth = _load_maybe_dict_npy(depth_path, "depth")

            # Map local emissivity IDs to global IDs if available
            if local_to_global is not None:
                local_ids = emap.astype(int)
                mapped = np.vectorize(local_to_global.get, otypes=[object])(local_ids)
                mapped_emap = np.where(mapped == None, 0, mapped).astype(int)  # fallback 0
            else:
                mapped_emap = emap.astype(int)

            # Save GT with original naming
            np.save(tmap_dir / f"tMap_{side}_{sample_id}.npy", tmap)
            np.save(emap_dir / f"eMap_{side}_{sample_id}.npy", emap)
            np.save(mapped_emap_dir / f"mapped_eMap_{side}_{sample_id}.npy", mapped_emap)
            np.save(depth_dir / f"Depth_{side}_{sample_id}.npy", depth)

            # Convert 54-band heatcube to 49 bands in microflicks (5th–53rd)
            heatcube_microflicks = W_m2_sr_cm1_to_microflicks(
                heatcube[:, :, 4:53],
                wavelengths_um=WAVELENGTHS_UM,
            )

            out_heatcube = hc_dir / f"HeatCube_{side}_{sample_id}.npy"
            np.save(out_heatcube, heatcube_microflicks[:, :, ::-1].astype(np.float32))
            print(f"[OK] {scene_dir.name} {side}_{sample_id} -> {out_heatcube}")


def main(root_dir: Path, out_root: Path | None = None) -> None:
    """
    Convert all scenes in root_dir from W/(m^2·sr·cm^-1) to microflicks,
    and save data as:
        out_root/<scene>/HeatCubes/HeatCube_{L|R}_XXXX.npy
        out_root/<scene>/GroundTruth/{tMap,eMap,mapped_eMap,Depth}/...
    """
    root_dir = root_dir.resolve()
    data_dir = out_root.resolve() if out_root is not None else root_dir.parent / "abdar"
    data_dir.mkdir(parents=True, exist_ok=True)

    for scene in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        scene_out = data_dir / scene.name
        process_scene(scene, scene_out)

    print(f"Created heatcubes with scene-specific structure in {data_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert HADAR heatcubes from W/(m^2·sr·cm^-1) to microflicks "
            "and reorganize GroundTruth + HeatCubes per scene."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/guillermo/ssd/datasets/raw/HADAR database"),
        help="Path to the original HADAR dataset root.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root directory. "
             "If omitted, a sibling 'abdar' folder is created next to data_root.",
    )

    args = parser.parse_args()
    main(root_dir=args.data_root, out_root=args.out_root)
