"""ERA5 dataset for training the conditional diffusion prior."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import xarray as xr
from torch.utils.data import Dataset

from data.dataloader_accelerate_miroc6 import (
    VAR_NAMES,
    get_DEM,
    get_lsm,
    time_to_features,
)


class ERA5PriorDataset(Dataset):
    """Load normalized ERA5 fields and the fixed geographic/time conditions.

    The input Zarr store must contain the ``z`` data variable, with dimensions
    ``time``, ``level``, latitude, and longitude.  The configured variables are
    selected using the common channel order defined in ``VAR_NAMES``.
    """

    def __init__(
        self,
        zarr_path: str,
        dem_path: str,
        lsm_path: str,
        mean_path: str,
        std_path: str,
        var_names: Sequence[str] = ("z250", "z500", "u10", "v10", "msl"),
        field_name: str = "z",
        time_dim: str = "time",
        level_dim: str = "level",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> None:
        required_paths = {
            "zarr_path": zarr_path,
            "dem_path": dem_path,
            "lsm_path": lsm_path,
            "mean_path": mean_path,
            "std_path": std_path,
        }
        missing = [name for name, path in required_paths.items() if not path]
        if missing:
            raise ValueError(f"Missing dataset paths: {', '.join(missing)}")

        self.var_names = list(var_names)
        unknown_vars = [name for name in self.var_names if name not in VAR_NAMES]
        if unknown_vars:
            raise ValueError(f"Unsupported variables: {', '.join(unknown_vars)}")
        self.variable_indices = np.asarray(
            [VAR_NAMES.index(name) for name in self.var_names], dtype=np.int64
        )

        era5 = xr.open_zarr(zarr_path)
        if field_name not in era5:
            raise KeyError(f"The ERA5 Zarr store must contain '{field_name}'.")
        fields = era5[field_name]
        missing_dims = [dim for dim in (time_dim, level_dim) if dim not in fields.dims]
        if missing_dims:
            raise ValueError(
                f"'{field_name}' must have dimensions {time_dim!r} and {level_dim!r}; "
                f"missing: {', '.join(missing_dims)}"
            )

        if start_time is not None or end_time is not None:
            fields = fields.sel({time_dim: slice(start_time, end_time)})
        fields = fields.isel({level_dim: self.variable_indices})

        spatial_dims = [dim for dim in fields.dims if dim not in (time_dim, level_dim)]
        if len(spatial_dims) != 2:
            raise ValueError(
                f"Expected two spatial dimensions after selecting '{field_name}', "
                f"got {spatial_dims}."
            )
        self.fields = fields.transpose(time_dim, level_dim, *spatial_dims)
        self.time_dim = time_dim
        self.height = self.fields.sizes[spatial_dims[0]]
        self.width = self.fields.sizes[spatial_dims[1]]
        if self.fields.sizes[time_dim] == 0:
            raise ValueError("The selected ERA5 time range contains no samples.")

        mean = np.asarray(np.load(mean_path), dtype=np.float32)[self.variable_indices]
        std = np.asarray(np.load(std_path), dtype=np.float32)[self.variable_indices]
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        if np.any(std == 0):
            raise ValueError("Normalization standard deviations must be nonzero.")
        try:
            np.broadcast_shapes(mean.shape, (len(self.var_names), self.height, self.width))
            np.broadcast_shapes(std.shape, (len(self.var_names), self.height, self.width))
        except ValueError as error:
            raise ValueError(
                "Normalization statistics are incompatible with the selected ERA5 fields."
            ) from error
        self.mean = mean
        self.std = std

        dem = get_DEM(dem_path).astype(np.float32, copy=False)
        lsm = get_lsm(lsm_path).astype(np.float32, copy=False)
        expected_shape = (self.height, self.width)
        if dem.shape != expected_shape or lsm.shape != expected_shape:
            raise ValueError(
                "DEM, land-sea mask, and ERA5 fields must use the same grid. "
                f"Expected {expected_shape}; got DEM {dem.shape} and LSM {lsm.shape}."
            )
        self.fixed_geography = np.ascontiguousarray(
            np.stack([dem, lsm], axis=0), dtype=np.float32
        )
        self.time_values = self.fields[time_dim].values

    def __len__(self) -> int:
        return len(self.time_values)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        timestamp = np.datetime_as_string(np.datetime64(self.time_values[index]), unit="s")
        image = self.fields.isel({self.time_dim: index}).values.astype(np.float32)
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))

        time_features = time_to_features(timestamp, self.height, self.width)
        condition = np.ascontiguousarray(
            np.concatenate([self.fixed_geography, time_features], axis=0), dtype=np.float32
        )
        return {"image": image, "condition": condition, "sample_id": timestamp}
