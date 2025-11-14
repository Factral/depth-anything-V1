from sklearn.decomposition import PCA
from typing import Union
import numpy as np  
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
from typing import Tuple


def get_rgb_image(cube, method):
    """
    Produce a 3-channel RGB visualization for the hyperspectral cube using:
    - 'sum': Sum across spectral bands, min-max normalize, replicate to RGB
    - 'pca': Z-score each band, PCA to 3 components, min-max normalize to [0,255] RGB
    Returns np.uint8 RGB image.
    """
    if method == 'sum':
        gray = cube.sum(axis=2)
        mn, mx = gray.min(), gray.max()
        gray_norm = (gray - mn) / (mx - mn + 1e-8)
        gray_255 = (gray_norm * 255).astype(np.uint8)
        # Return RGB = BGR. They are the same because the channels are the same?
        return np.stack([gray_255, gray_255, gray_255], axis=-1)
    elif method == 'pca':
        H, W, C = cube.shape
        flat = cube.reshape(-1, C).astype(np.float32)
        # Z-score standardization per band so all bands contribute
        mu = flat.mean(axis=0, keepdims=True)
        sigma = flat.std(axis=0, keepdims=True)
        flat_std = (flat - mu) / sigma
        pcs = PCA(n_components=3).fit_transform(flat_std).reshape(H, W, 3)
        mn, mx = pcs.min(), pcs.max()
        rgb_norm = (pcs - mn) / (mx - mn + 1e-8)
        rgb_255 = (rgb_norm * 255).astype(np.uint8)
        return rgb_255
    else:
        raise ValueError(f"Unknown method: {method}")


def save_image(img, path, cmap=None, vmin=0, vmax=None):
    """
    Save an image to the specified path.
    
    Args:
        img: Image data
        path: Path to save the image
        cmap: Colormap to use (default: None)
    """
    plt.figure(figsize=(8, 6))
    # Assume inputs are RGB now (no implicit channel flipping)
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


def _load_maybe_dict_npy(path: str, key_fallback: str = None, # type: ignore
                         dtype:Union[np.float32, np.uint8] = np.float32): # type: ignore
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            maybe = arr.item()
            if isinstance(maybe, dict):
                arr = maybe
        except Exception:
            pass
    if isinstance(arr, dict):
        if key_fallback is None:
            # return first value
            return next(iter(arr.values()))
        return arr.get(key_fallback, next(iter(arr.values())))
    return arr.astype(dtype)


def _local_to_global_mapping(elist_path: str):
    elist_raw = np.load(elist_path, allow_pickle=True)
    if isinstance(elist_raw, np.ndarray) and elist_raw.dtype == object:
        try:
            maybe = elist_raw.item()
            if isinstance(maybe, dict):
                elist_raw = maybe
        except Exception:
            pass
    if isinstance(elist_raw, dict):
        elist = elist_raw.get('eList', next(iter(elist_raw.values())))
    else:
        elist = elist_raw
    elist = np.asarray(elist).reshape(-1)

    # Auto-detect 1-based elist
    one_based_elist = (elist.min() >= 1) and (0 not in elist)
    base_offset = 1 if one_based_elist else 0

    # Keys are 1..len(elist) to match 1-based local indices in eMap
    local_to_global = {int(loc + 1): int(elist[loc] - base_offset) for loc in range(len(elist))}
    return local_to_global


def _obtain_spectral_resolution(lambda_vals: np.ndarray) -> int:
    """
    Args:
        lambda_vals: Wavelenghts values in micrometers.

    Returns:
        resolution: Spectral resolution in nanometers
    """
    return int(np.round(np.mean(np.diff(lambda_vals)) * 1e3))


def adjust_spectral_data(
    data_array: np.ndarray,
    lambda_vals_desired: np.ndarray,
    resolution: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust the spectral data to match the desired wavelength grid and resolution.

    Parameters:
        data_array (np.ndarray): Original data array with wavelengths and values.
        lambda_vals_desired (np.ndarray): Desired wavelength grid.
        resolution (int): Resolution of the wavelength grid in nanometers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted data array and matched wavelength grid.
    """
    data = data_array[:, 1:]  # Shape: (N_wavelengths, N_columns)

    # It represents the number of data points within the specified resolution in micrometers. 
    w_std = resolution * 1e-3 / np.mean(np.diff(data_array[:, 0]))
    alpha = len(data_array) / w_std
    
    # Calculate the standard deviation for the Gaussian window
    # Reference: https://stackoverflow.com/questions/71196448/why-does-scipy-signal-windows-gaussian-produce-different-numbers-than-matlab-and
    std = (len(data_array) - 1) / (2 * alpha)
    w = gaussian(len(data_array), std=std)
    w /= np.sum(w)

    pad_width = 2 * int(np.ceil(w_std))
    adjusted_data = np.pad(data, ((pad_width, pad_width), (0, 0)), mode='constant', constant_values=1)

    w = np.pad(w, pad_width, mode='constant')

    # Convolve each column of the data array with the Gaussian window
    for i in range(data.shape[1]):
        convolved_data = fftconvolve(adjusted_data[:, i], w, mode='same')
        adjusted_data[:, i] = convolved_data

    wavelength_padded = np.pad(
        data_array[:, 0], pad_width, mode='linear_ramp',
        end_values=(
            data_array[0, 0] - pad_width * np.mean(np.diff(data_array[:, 0])),
            data_array[-1, 0] + pad_width * np.mean(np.diff(data_array[:, 0]))
        )
    )

    # Find the corresponding indices using np.searchsorted, this is 10x more efficient than argmin
    index = np.searchsorted(wavelength_padded, lambda_vals_desired)
    index = np.clip(index, 0, len(wavelength_padded) - 1)

    # Obtain the adjusted values using advanced indexing
    adjusted_data = adjusted_data[index + pad_width]
    lambda_vals_matched = wavelength_padded[index]

    return adjusted_data, lambda_vals_matched


def _compute_attenuation(transmittance, desired_attenuation_units):
    """
    Compute the attenuation from a transmittance signature. This assumes an air 
    column of 1m. 

    Args:
        transmittance: Array with the transmittance values across the spectrum.
        desired_attenuation_units: Units on which we want the attenuation.

    Returns:
        attenuation (np.array): Attenuation in log10_per_m units.
        attenuation_units (str): The units of the attenuation.
    """
    if desired_attenuation_units == 'dB_per_m':
        attenuation = - 10 * np.log10(transmittance).squeeze()
        attenuation_units = 'dB_per_m'
    elif desired_attenuation_units == 'log10_per_m':
        attenuation = - np.log10(transmittance).squeeze()
        attenuation_units = 'log10_per_m'
    else:
        raise NotImplementedError('There are no more options for the units. (to' \
        'my knowledge)')
    return {'attenuation': attenuation, 
            'attenuation_units': attenuation_units}


def _compute_transmittance(attenuation, depth, attenuation_units) -> np.ndarray:
    """
    Compute the transmittance of the scene witth the attenuation and the ground
    truth depth. Using the following tho possible equations

        attenuation_units = 'dB_per_m'    : 10^{ - α(λ) d / 10 }
        attenuation_units = 'log10_per_m' : 10^{ - α(λ) d }

    Args:
        attenuation: Assumed attenuation.
        depth: Ground truth depth of the scene.
        attenuation_units: Units of the attenuation. It can be dB_per_m or log10_per_m.

    Returns:
        transmittance (np.ndarray): Transmittance function of the scene. 
    """
    if attenuation_units == 'dB_per_m':
        transmittance = 10.0 ** ((-attenuation[None, None, :, 0] * depth[..., None]) / 10.0)
    elif attenuation_units == 'log10_per_m':  # 'log10_per_m'
        transmittance = 10.0 ** (-attenuation[None, None, :, 0] * depth[..., None])
    else:
        raise NotImplementedError("Incompatible units. Check them.")
    # Ensure (H, W, C)
    transmittance = transmittance.reshape(transmittance.shape[0], transmittance.shape[1], -1)
    return transmittance
    

def _load_hadar_emissivity(mapped_emap, emissivity_file=None):
    """
    Load the emissivity signatures from the HADAR Database.

    Args:
        mapped_emap: Emissivity map, mapped from local indexes to global ones. 
            It has to be an Image obbject of shapes [H, W, C]. And we take one
            channel because the other ones contains the same information.
        emissivity_file (str): Path to the file with the emissivities of the 
            database.
    
    Returns:
        emissivity: A cube with the per-pixel emissivity. It should have shapes
            [H, W, N], where N is the number of bands in the emissivity signatures.
    """

    if emissivity_file is None:
        emissivity_file = 'data/local_npys/matLib_FullDatabase.npy'

    # Load and adjust emissivity data for HADAR
    emissivity_data = np.load(emissivity_file, allow_pickle=True)
    emissivity_data = emissivity_data.item().get('matLib').astype(np.float32).transpose()  # Shape: (N_materials, N_wavenumbers)
    emissivity_data = emissivity_data[:, ::-1] # We have to reverse because original emissivity comes in wavenumbers (cm⁻1)

    return emissivity_data[np.array(mapped_emap)[:, :, 0]]  # Shape: (H, W, N_wavelengths) or (N_wavelengths,)