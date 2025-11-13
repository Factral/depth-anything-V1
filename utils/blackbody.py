import numpy as np
from scipy.constants import Planck as h, speed_of_light as c, Boltzmann as k
import matplotlib.pyplot as plt
from typing import Union, Tuple


def blackbody(lambda_vals: np.ndarray,
              T: Union[float, np.ndarray],
              units: str = 'watts',
              return_derivative: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate blackbody radiance (Planck) and optionally dB/dT.

    Parameters:
        lambda_vals (np.array): Wavelengths in micrometers.
        T (float or np.ndarray): Temperature in Kelvin (scalar or array, broadcastable).
        units (str): 'watts' (W/m^2/sr/m) or 'microflicks' (μW/cm^2/sr/μm).
        return_derivative (bool): If True return (B, dB/dT). Otherwise only B.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            B (and dB/dT if return_derivative=True) in requested units.
    """
    T = np.asarray(T, dtype=np.float32)

    lam_m = np.array(lambda_vals, dtype=np.float32) * 1e-6
    lam5 = lam_m**5
    C1 = 2 * h * c**2
    C2 = h * c / k
    x = C2 / (lam_m * T)
    ex = np.exp(x)
    denom = ex - 1.0
    B_watts = C1 / (lam5 * denom)

    if return_derivative:
        # dB/dT = C1 * ex * x / (T * lam^5 * (ex - 1)^2)
        dB_dT_watts = C1 * ex * x / (T * lam5 * denom**2)

    if units == 'microflicks':
        scale = 1e-4  # W/m^2/sr/m -> μW/cm^2/sr/μm
        B = B_watts * scale
        if return_derivative:
            dB_dT = dB_dT_watts * scale
    elif units == 'watts':
        B = B_watts
        if return_derivative:
            dB_dT = dB_dT_watts
    else:
        raise ValueError("units must be 'watts' or 'microflicks'")

    if return_derivative:
        return B, dB_dT
    return B


def plot_propagation(radiance: np.ndarray, lambda_vals: np.ndarray, pixel: Tuple[int, int] = (0, 0), units: str = 'watts'):
    """
    Plot the spectral radiance data for a specific pixel.

    Parameters:
    radiance (np.array): Spectral radiance data.
    lambda_vals (np.array): Wavelength values.
    pixel (tuple): Pixel coordinates to plot.
    units (str): Units of the radiance ('watts' or 'microflicks').
    """
    if radiance.ndim == 3:
        data_to_plot = radiance[pixel[0], pixel[1], :]
    else:
        data_to_plot = radiance

    plt.figure(figsize=(15, 6))
    plt.plot(lambda_vals.squeeze(), data_to_plot)
    plt.xlabel('Wavelength (micrometers)')
    if units == 'microflicks':
        plt.ylabel('Spectral Radiance (μW/cm²/sr/μm)')
    else:
        plt.ylabel('Spectral Radiance (W/m²/sr/m)')
    plt.title(f'Propagation at pixel {pixel}')
    plt.show()


def microflicks_to_W_m2_sr_m(L_microflicks):
    """Convert radiance from microflick units (μW/cm²/sr/μm) to W/m²/sr/m.
    Factor derivation:
      μW -> W        : 1e-6
      /cm² -> /m²    : 1e4
      per μm -> per m: 1e6
      Total factor   : 1e-6 * 1e4 * 1e6 = 1e4
    So: L_W_m2_sr_m = L_microflicks * 1e4
    """
    return L_microflicks * 1e4


def W_m2_sr_m_to_microflicks(L_watts_per_m):
    """
    Convert radiance from W/m^2/sr/m to microflicks (μW/cm^2/sr/μm).
    Factor: 1e-4 (see microflicks_to_W_m2_sr_m for derivation).
    """
    return np.asarray(L_watts_per_m) * 1e-4


def cm1_to_um(wavenumbers_cm1):
    """
    Convert wavenumber (cm^-1) to wavelength (μm).
    λ(μm) = 1e4 / σ(cm^-1)
    """
    nu = np.asarray(wavenumbers_cm1, dtype=float)
    return 1e4 / nu


def W_m2_sr_cm1_to_microflicks(L_per_cm1, wavelengths_um):
    """
    Convert spectral radiance from W/(m^2·sr·cm^-1) to microflicks (μW/cm^2·sr/μm).
    Uses Jacobian: L_λ = L_σ * |dσ/dλ| with σ in cm^-1, λ in meters.
      |dσ/dλ_m| = 1/(100 * λ_m^2)  and then W/m^2/sr/m -> microflicks via 1e-4.
    Result factor: 1e6 / λ_μm^2
    """
    L = np.asarray(L_per_cm1, dtype=float)
    lam_um = np.asarray(wavelengths_um, dtype=float).ravel()
    if L.shape[-1] != lam_um.shape[0]:
        raise ValueError(f"Spectral length mismatch: cube B={L.shape[-1]} vs wavelengths={lam_um.shape[0]}")
    factor = 1e6 / (lam_um**2)  # shape (B,)
    return L * factor


def microflicks_to_W_m2_sr_cm1(L_microflicks, wavelengths_um):
    """
    Inverse of W_m2_sr_cm1_to_microflicks.
    Convert microflicks (μW/cm^2·sr/μm) to W/(m^2·sr·cm^-1).
    Inverse factor: λ_μm^2 / 1e6
    """
    L = np.asarray(L_microflicks, dtype=float)
    lam_um = np.asarray(wavelengths_um, dtype=float).ravel()
    if L.shape[-1] != lam_um.shape[0]:
        raise ValueError(f"Spectral length mismatch: cube B={L.shape[-1]} vs wavelengths={lam_um.shape[0]}")
    factor_inv = (lam_um**2) / 1e6  # shape (B,)
    return L * factor_inv


def _to_W_m2_sr_m(L: np.ndarray, units: str) -> np.ndarray:
    """
    Convert to W/(m^2·sr·m) to match blackbody(..., units='watts').
    """
    u = units.lower()
    if u in ("microflicks", "μw/cm^2/sr/μm", "uw/cm^2/sr/um"):
        return microflicks_to_W_m2_sr_m(np.asarray(L, dtype=np.float64))
    if u in ("w_m2_sr_m", "w/m^2/sr/m", "watts", "si"):
        return np.asarray(L, dtype=np.float64)
    raise ValueError(f"Unsupported units for radiance: {units}")


def planck_inverse_lambda(L, lam_m):
    """Invert Planck (wavelength form) for L in W / m^2 / sr / m.
    Returns temperature (K).
    """
    L = np.maximum(L, 1e-30)
    a = 2 * h * c**2
    b = h * c / k
    return b / (lam_m * np.log1p(a / ((lam_m**5) * L)))


def brightness_temperature_um(
    wavelength_um: np.ndarray, radiance: np.ndarray, units: str = "microflicks"
) -> np.ndarray:
    """
    Invert Planck to brightness temperature T [K] at wavelength_um (µm).
    radiance: per-µm (microflicks) or per-m (SI). Converted to W/m^2/sr/m internally.
    """
    lam_m = np.asarray(wavelength_um, dtype=np.float64) * 1e-6
    L_watts = _to_W_m2_sr_m(radiance, units=units)  # W/(m^2·sr·m)
    # planck_inverse_lambda expects W/(m^2·sr·m) and λ in meters
    T = planck_inverse_lambda(L_watts, lam_m)
    return np.where(np.isfinite(T), T, 0.0)


def compute_bt_cube(hsi_cube: np.ndarray, wavelengths_um: np.ndarray, units: str = "microflicks") -> np.ndarray:
    """
    Compute brightness temperature per pixel per band.
    hsi_cube: (H,W,B)
    returns BT_cube: (H,W,B) in Kelvin
    """
    H, W, B = hsi_cube.shape
    BT = np.empty_like(hsi_cube, dtype=np.float64)
    for b in range(B):
        BT[..., b] = brightness_temperature_um(wavelengths_um[b], hsi_cube[..., b], units=units)
    return BT


def estimate_air_temperature_from_band(
    cube,
    wavelengths_um,
    lambda_sat,
    window_um=0.02,
    mask=None,
    aggregate="median",
    per_pixel=False,
    radiance_units="microflicks",  # 'microflicks' or 'W_m2_sr_m'
    custom_scale=None              # override full conversion factor
):
    """Estimate air temperature using saturated band inversion.
    cube units:
      - If radiance_units='microflicks' (μW/cm²/sr/μm) auto-converted to W/m²/sr/m.
      - If 'W_m2_sr_m' assumed already in W/m²/sr/m.

    Parameters:
      cube            : (H,W,B) spectral radiance cube (must be in physical radiance units after radiance_scale)
      wavelengths_um  : 1-D array length B (micrometers)
      lambda_sat      : target saturated wavelength (micrometers)
      window_um       : half-width total selection window (bands with |λ-λ_sat| <= window/2)
      mask            : optional boolean (H,W) selecting pixels to use (True=use)
      aggregate       : 'median' or 'mean' for global estimate
      per_pixel       : if True also return per-pixel temperature map (using local band radiance)
      radiance_scale  : factor to multiply cube radiance values to convert to W/sr/m^3 units
                        (If cube already calibrated properly, leave as 1.0)
    
    Returns:
      T_air_global, (optional T_air_map if per_pixel=True)
    """
    wavelengths_um = np.asarray(wavelengths_um).ravel()
    cube = np.asarray(cube)
    H, W, B = cube.shape
    if wavelengths_um.shape[0] != B:
        raise ValueError("wavelengths length mismatch")

    diff = np.abs(wavelengths_um - lambda_sat)
    band_sel = np.where(diff <= window_um / 2)[0]
    if band_sel.size == 0:
        band_sel = np.array([np.argmin(diff)])

    # Average selected bands
    L_sel_raw = cube[..., band_sel].mean(axis=-1)  # (H,W) in input units

    if mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        mask = mask.astype(bool)

    # Determine conversion scale
    if custom_scale is not None:
        scale = float(custom_scale)
    else:
        if radiance_units == "microflicks":
            scale = 1e4  # via derivation above
        elif radiance_units == "W_m2_sr_m":
            scale = 1.0
        else:
            raise ValueError("radiance_units must be 'microflicks' or 'W_m2_sr_m'")

    L_sel = L_sel_raw * scale  # now W/m²/sr/m

    lam_use_m = wavelengths_um[band_sel].mean() * 1e-6
    L_valid = L_sel[mask]
    if L_valid.size == 0:
        raise ValueError("No valid pixels under mask")

    T_valid = planck_inverse_lambda(L_valid, lam_use_m)
    if aggregate == "median":
        T_air_global = float(np.median(T_valid))
    elif aggregate == "mean":
        T_air_global = float(np.mean(T_valid))
    else:
        raise ValueError("aggregate must be 'median' or 'mean'")

    if not per_pixel:
        return T_air_global

    T_map = np.full((H, W), np.nan, dtype=float)
    T_map[mask] = T_valid
    return T_air_global, T_map


# Helper: auto-pick saturated lambda (minimum transmittance) from attenuation array
def select_lambda_sat(wavelengths_um, transmittance):
    idx = np.argmin(transmittance)
    return float(wavelengths_um[idx])