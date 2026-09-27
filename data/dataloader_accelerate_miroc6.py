
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import xarray as xr
from datetime import datetime
import rasterio
import cv2
from scipy.ndimage import zoom
from util.logger import get_logger
import torch

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str = None, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool,
                   sampler: Optional[torch.utils.data.Sampler] = None):
    shuffle = train and (sampler is None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
        sampler=sampler,
    )

def get_DEM(file_path):
    with rasterio.open(file_path) as dataset:
        band1 = dataset.read(1)
        band1 = cv2.resize(band1,(1440,720))

        transform = dataset.transform
        width = dataset.width
        height = dataset.height
        pixel_size = transform[0]
        west = transform[2]
        east = west + width * pixel_size

        split_index = int(width / 2)
        band1_shifted = np.hstack((band1[:, split_index:], band1[:, :split_index]))
        band1_shifted = (band1_shifted - band1_shifted.mean()) / band1_shifted.std()
    return band1_shifted


def get_lsm(file_path):
    ds = xr.open_dataset(file_path)
    lsm = ds['lsm'].values
    lsm_720 = zoom(lsm, (720/721, 1440/1440), order=1)

    return lsm_720


def time_to_features(timestamp, height, width):
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    month = dt.month
    day = dt.day
    hour = dt.hour

    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    day_sin = np.sin(2 * np.pi * day / 31.0)
    day_cos = np.cos(2 * np.pi * day / 31.0)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    time_features = np.array(
        [month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos],
        dtype=np.float32
    ).reshape(6, 1, 1)
    time_features = np.tile(time_features, (1, height, width))

    return time_features

RESET_SEQ = "\033[0m"
RED_SEQ = "\033[31m"
GREEN_SEQ = "\033[32m"
VAR_NAMES = ['z250', 'z500', 'z700', 'z850', 't250', 't500', 't700', 't850', 'u250', 'u500',
 'u700', 'u850', 'v250', 'v500', 'v700', 'v850', 't2m', 'u10', 'v10', 'msl',"tp", "sst"]


@register_dataset(name='zarr_climate')
class ZarrClimateDataset(Dataset):
    def __init__(
        self,
        flag='test',
        var_names=("z250", "z500", "u10", "v10", "msl"),
        root=None,
        zarr_path=None,
        dem_path=None,
        lsm_path=None,
        mean_path=None,
        std_path=None,
        start_time=None,
        end_time=None,
        **kwargs,
    ):
        logger = get_logger()
        start_time_total = time.time()
        self.flag = flag
        self.var_names = list(var_names)

        required_paths = {
            'zarr_path': zarr_path,
            'dem_path': dem_path,
            'lsm_path': lsm_path,
            'mean_path': mean_path,
            'std_path': std_path,
        }
        missing = [name for name, path in required_paths.items() if not path]
        if missing:
            raise ValueError(
                f"Missing dataset paths in the task config: {', '.join(missing)}"
            )
        logger.info("Loading MIROC6 downscaling dataset.")

        # ==================== 1. Load constants and Zarr/NetCDF ====================
        time_step = time.time()
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        logger.info(f"Loaded normalization statistics in {time.time() - time_step:.2f}s.")
        time_step = time.time()

        miroc6_data = xr.open_zarr(zarr_path)

        logger.info(f"Opened MIROC6 Zarr in {time.time() - time_step:.2f}s.")
        time_step = time.time()

        self.global_dem = np.expand_dims(get_DEM(dem_path).astype(np.float32), axis=0)
        self.global_lsm = np.expand_dims(get_lsm(lsm_path).astype(np.float32), axis=0)
        logger.info(f"Loaded DEM and land-sea mask in {time.time() - time_step:.2f}s.")
        time_step = time.time()

        if start_time and end_time:
            self.time_range = [start_time, end_time]
        elif flag == "train":
            self.time_range = ["1980-01-02","2000-12-31"]  # Out of MIROC6 range, xarray will handle
        elif flag == "val":
            self.time_range = ["2001-01-01","2001-01-02"]  # Use MIROC6 actual time
        elif flag == "test":
            self.time_range = ["2001-01-01","2001-12-31"]  # Adjusted to MIROC6 actual time range


        date_range = slice(self.time_range[0], self.time_range[1])
        miroc6_data = miroc6_data.sel(time=date_range)
        logger.info(f"Selected {self.time_range[0]} to {self.time_range[1]} in {time.time() - time_step:.2f}s.")
        time_step = time.time()

        # ==================== 4. Prepare normalization parameters (MIROC6 variable indices) ====================
        # MIROC6's 5 variables: z250, z500, u10, v10, msl
        # Corresponding indices in VAR_NAMES: [0, 1, 17, 18, 19]
        self.var_index = np.array([VAR_NAMES.index(var) for var in self.var_names])

        if 'zg' not in miroc6_data:
            raise KeyError("The MIROC6 Zarr store must contain a 'zg' variable.")
        self.label_dataset = miroc6_data['zg']
        logger.info(f"Prepared lazy Xarray selections in {time.time() - time_step:.2f}s.")
        time_step = time.time()

        # ==================== 5. Generate target lat/lon coordinates (720x1440, 0.25 degree resolution) ====================
        # Target grid: ERA5's 0.25 degree grid centers
        # Latitude: 90 to -90, 720 points, center aligned
        self.target_lat = np.linspace(90, -90, 720, endpoint=False)
        # Longitude: 0 to 360, 1440 points, center aligned
        self.target_lon = np.linspace(0, 360, 1440, endpoint=False)

        # ==================== 6. Get length ====================
        self.time_list = self.label_dataset['time'].values
        self.dataset_len = len(self.time_list)
        logger.info(f"Dataset contains {self.dataset_len} time steps.")

        end_time_total = time.time()
        execution_time_total = end_time_total - start_time_total

        logger.info(f"Loaded MIROC6 dataset in {execution_time_total:.2f}s.")


    def __getitem__(self, index):

        sample_id = None
        label = None

        sample_id = str(np.datetime64(self.time_list[index], 's'))
        time_feature = time_to_features(sample_id, 720, 1440)

        # Select the requested time step.
        label_miroc6 = self.label_dataset.sel(time=self.time_list[index])

        # Normalize with DataArray objects to preserve channel alignment.
        mean_da = xr.DataArray(self.mean[self.var_index], dims=['channel'])
        std_da = xr.DataArray(self.std[self.var_index], dims=['channel'])
        label_normalized = (label_miroc6 - mean_da) / std_da

        # =========================================================
        # [Fix] Resolve interpolation boundary black edges for 0-360 full coverage data
        # =========================================================

        # Since the data already includes 0.0 and 360.0, no concat is needed.
        # Key is to use kwargs={'fill_value': 'extrapolate'}.
        # It handles target grid points like 359.875 that may be judged as "slightly out of bounds" due to precision differences.


        # Coarsen the source grid, then interpolate the observation to the target grid.
        lat_factor = round(128 / 36)
        lon_factor = round(256 / 72)
        label_downsampled = label_normalized.coarsen(lat=lat_factor, lon=lon_factor, boundary='trim').mean()

        label_upsampled = label_downsampled.interp(
            lat=self.target_lat,
            lon=self.target_lon,
            method='linear',
            # Extrapolation avoids small floating-point gaps at the grid boundary.
            kwargs={'fill_value': 'extrapolate'}
        )

        # Convert to NumPy and replace any residual missing values.
        label = label_upsampled.values.astype(np.float32)
        label = np.nan_to_num(label, nan=0.0)

        condition = np.concatenate([self.global_dem, self.global_lsm, time_feature], axis=0)

        return {
            'image': label,
            'condition': condition,
            'sample_id': sample_id
        }
        
    def __len__(self):
        return self.dataset_len
