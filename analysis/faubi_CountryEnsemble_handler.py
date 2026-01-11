# faubi_CountryEnsemble_handler.py
from __future__ import annotations

import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio.v3 as iio

try:
    import dask  # type: ignore
    dask.config.set({"array.slicing.split_large_chunks": True})
except Exception:
    pass


@dataclass
class FAUBIEnsemble:
    """
    Analyzer for FAUBI ensemble runs (livestock x country) for a given year.
    - Robust discovery (deep scan under **/OutputDir)
    - Lazy/streaming loading and corruption skipping
    - Tile-based spot checks (memory-safe)
    - Diffs vs reference and GIF generation
    """
    ensemble_base_dir: str
    reference_dir: str
    year: int = 2015

    # Country ensembles: provide CSVs to define the country list and (optionally) the expected runs.
    # - key_countries_csv should contain at least: Country, Short_Name (and optionally MaskNo/MaskName).
    # - run_manifest_csv should contain at least: RUNID, LIVESTOCK, COUNTRY (and optionally MASK).
    key_countries_csv: Optional[str] = None
    run_manifest_csv: Optional[str] = None

    # Files look like: GEOSChem.AerosolMass.2015MM01_0000z.nc4 (MM=01..12)
    filename_glob: str = "GEOSChem.AerosolMass.*{year}[0-1][0-9]01_0000z.nc4"

    livestock_map: Dict[str, str] = field(default_factory=lambda: {
        "Bf": "BUFFALO", "Ch": "CHICKEN", "Ct": "CATTLE", "Dk": "DUCK",
        "Gt": "GOAT", "Ho": "HORSE", "Pg": "PORK", "Sh": "SHEEP",
    })
    countries: List[str] = field(default_factory=list)  # populated in __post_init__ or provided explicitly
    country_map: Dict[str, str] = field(default_factory=dict)  # short_name -> friendly label
    run_manifest: Optional[object] = None  # pandas.DataFrame if loaded

    # Loaded artifacts
    ensemble_ds: Optional[xr.Dataset] = None
    reference_ds: Optional[xr.Dataset] = None
    diff_da: Optional[xr.DataArray] = None  # PM25 deltas vs ref, lev=0

    # Discovery cache
    _scenarios: Optional[List[Tuple[str, str, str]]] = None  # [(run_dir, livestock_code, country)]

    # Diagnostics
    skipped_files: Dict[str, List[str]] = field(default_factory=dict)  # run_dir -> list of corrupted files

    def __post_init__(self) -> None:
        """Populate country metadata from CSVs if provided (or if default filenames exist)."""
        # Load Key_Countries.csv to define the country axis ordering (if countries not explicitly set).
        if not self.countries:
            csv_path = self.key_countries_csv
            if csv_path is None:
                # Try a sensible default relative to the working directory
                if Path('Key_Countries.csv').exists():
                    csv_path = 'Key_Countries.csv'
            if csv_path is not None and Path(csv_path).exists():
                try:
                    import pandas as pd
                    key = pd.read_csv(csv_path)
                    # Prefer Short_Name if present (directory-friendly), else fall back to Country.
                    if 'Short_Name' in key.columns:
                        self.countries = [str(x) for x in key['Short_Name'].tolist()]
                    elif 'Country' in key.columns:
                        self.countries = [str(x) for x in key['Country'].tolist()]
                    else:
                        raise ValueError(f"{csv_path} must contain a 'Short_Name' or 'Country' column")
                    # Keep a map for friendly labeling if both columns exist
                    if 'Country' in key.columns and 'Short_Name' in key.columns:
                        self.country_map = {str(sn): str(cn) for cn, sn in zip(key['Country'], key['Short_Name'])}
                    else:
                        self.country_map = {c: c for c in self.countries}
                except Exception as e:
                    warnings.warn(f"[FAUBIEnsemble] Failed to read key_countries_csv={csv_path}: {repr(e)}")
                    self.countries = []
                    self.country_map = {}
            else:
                self.country_map = {c: c for c in self.countries}

        # Load run manifest if provided (used only for ordering / sanity checks, not required).
        self.run_manifest = None
        mf_path = self.run_manifest_csv
        if mf_path is None and Path('Country_Run_Manifest.csv').exists():
            mf_path = 'Country_Run_Manifest.csv'
        if mf_path is not None and Path(mf_path).exists():
            try:
                import pandas as pd
                self.run_manifest = pd.read_csv(mf_path)
            except Exception as e:
                warnings.warn(f"[FAUBIEnsemble] Failed to read run_manifest_csv={mf_path}: {repr(e)}")

    # ------------------------- Discovery -------------------------

    def _parse_scenario_from_dirname(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse (livestock_code, country) from a path string using lenient token matching.
        """
        lc = None
        pol = None
        for code in self.livestock_map.keys():
            if re.search(rf"(^|[_\-/]){code}([_\-/]|$)", name):
                lc = code
                break
        for p in self.countries:
            if re.search(rf"(^|[_\-/]){p}([_\-/]|$)", name):
                pol = p
                break
        return lc, pol

    def discover(self, *, strict_labels: bool = False) -> List[Tuple[str, str, str]]:
        """
        Deep-scan the ensemble base dir for monthly outputs.
        Returns (run_dir, livestock_code_or_scenario, country_or_unknown).

        - Searches for any '**/OutputDir' and matches files with self.filename_glob
        - Parses (livestock, country) from FULL PATH (lenient).
        - If parsing fails and strict_labels=False, assigns a unique scenario id (SCN###) and 'UNK'.
        """
        base = Path(os.path.expanduser(self.ensemble_base_dir))
        if not base.exists():
            raise FileNotFoundError(f"Ensemble base dir not found: {base}")

        output_dirs = [p for p in base.rglob("OutputDir") if p.is_dir()]
        pattern = self.filename_glob.format(year=self.year)

        scenarios: List[Tuple[str, str, str]] = []
        scenario_counter = 0

        for odir in sorted(output_dirs):
            files = sorted(glob.glob(str(odir / pattern)))
            if not files:
                continue

            full_path_str = str(odir.parent)  # run dir is parent of OutputDir
            lc, pol = self._parse_scenario_from_dirname(full_path_str)

            if lc is None or pol is None:
                if strict_labels:
                    # Skip unlabeled runs in strict mode
                    continue
                scenario_counter += 1
                lc = lc or f"SCN{scenario_counter:03d}"
                pol = pol or "UNK"

            scenarios.append((str(odir.parent), lc, pol))

        if not scenarios:
            raise RuntimeError(
                f"No runs discovered under {base}. "
                f"Looked for '**/OutputDir/{pattern}'. "
                f"Adjust 'filename_glob' or set strict_labels=False."
            )

        self._scenarios = scenarios
        return scenarios

    # ------------------------- File utilities -------------------------

    def _glob_monthlies(self, run_dir: str) -> List[str]:
        """Return list of monthly files for a run (may include corrupt ones)."""
        pattern = self.filename_glob.format(year=self.year)
        return sorted(glob.glob(str(Path(run_dir) / "OutputDir" / pattern)))

    def _filter_corrupt_files(self, files: List[str]) -> List[str]:
        """
        Probe each file with a quick open/close to weed out corrupt HDF/NetCDF files.
        Records any skipped files into self.skipped_files.
        """
        good, bad = [], []
        for f in files:
            try:
                # Fast integrity check: open/close without decoding times
                with xr.open_dataset(f, engine="netcdf4", decode_times=False, chunks={}) as _:
                    pass
                good.append(f)
            except Exception as e:
                bad.append((f, repr(e)))

        if bad:
            run_key = str(Path(files[0]).parents[1]) if files else "UNKNOWN_RUN"
            self.skipped_files.setdefault(run_key, []).extend([p for p, _ in bad])
            msgs = "\n".join([f"{p}  <-- {err}" for p, err in bad])
            warnings.warn(
                f"[FAUBIEnsemble] Skipping {len(bad)} corrupt/failed file(s) in run {run_key}:\n{msgs}"
            )
        return good

    # ------------------------- Openers -------------------------

    def _open_monthlies_streaming(
        self,
        run_dir: str,
        keep_vars: Optional[List[str]] = None,
        drop_vars: Optional[List[str]] = None,
        chunks: Optional[Dict] = None,
        decode_times: bool = True,
    ) -> xr.Dataset:
        """
        Per-file lazy open + concat across time with minimal metadata reconciliation.
        Avoids open_mfdataset RAM blowups. Skips corrupt files.
        """
        files = self._glob_monthlies(run_dir)
        if not files:
            raise FileNotFoundError(
                f"No files under {run_dir}/OutputDir matching: {self.filename_glob.format(year=self.year)}"
            )
        files = self._filter_corrupt_files(files)
        if not files:
            raise RuntimeError(f"All monthly files unreadable in {run_dir}")

        opened = []
        for f in files:
            try:
                ds = xr.open_dataset(
                    f,
                    engine="netcdf4",
                    decode_times=decode_times,
                    mask_and_scale=True,
                    chunks=chunks or {"time": 1},
                    cache=False,
                    lock=False,
                )
                if keep_vars is not None:
                    present = [v for v in keep_vars if v in ds.data_vars]
                    to_drop = [v for v in ds.data_vars if v not in present]
                    if to_drop:
                        ds = ds.drop_vars(to_drop)
                elif drop_vars is not None:
                    to_drop = [v for v in drop_vars if v in ds.data_vars]
                    if to_drop:
                        ds = ds.drop_vars(to_drop)
                opened.append(ds)
            except Exception as e:
                warnings.warn(f"[FAUBIEnsemble] Skipping file (failed to open): {f}  <-- {repr(e)}")

        if not opened:
            raise RuntimeError(f"No usable monthly files in {run_dir}")

        ds = xr.concat(
            opened,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

        if "time" in ds:
            ds = ds.sortby("time")

        # Close underlying file handles (graphs remain)
        for src in opened:
            try:
                src.close()
            except Exception:
                pass

        return ds

    def _open_monthlies(self, run_dir: str) -> xr.Dataset:
        """
        Fallback opener using open_mfdataset (kept as a reference). Skips corrupt files first.
        """
        files = self._glob_monthlies(run_dir)
        if not files:
            raise FileNotFoundError(
                f"No files under {run_dir}/OutputDir matching: {self.filename_glob.format(year=self.year)}"
            )
        files = self._filter_corrupt_files(files)
        if not files:
            raise RuntimeError(f"All monthly files unreadable in {run_dir}")

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            parallel=True,
            engine="netcdf4",
        )
        if "time" in ds:
            ds = ds.sortby("time")
        return ds

    # ------------------------- Loaders -------------------------

    def load_ensemble(
        self,
        *,
        streaming: bool = True,
        keep_vars: Optional[List[str]] = None,  # e.g., ["PM25", "NH3", "NOx", "VOC"]
        drop_vars: Optional[List[str]] = None,
        chunks: Optional[Dict] = None,          # e.g., {"time": 1}
        decode_times: bool = True,
        strict_labels: bool = False,
    ) -> xr.Dataset:
        """
        Discover scenarios and build the combined dataset.
        """
        scenarios = self._scenarios or self.discover(strict_labels=strict_labels)
        datasets = []
        skipped_runs = []

        for run_dir, lc, pol in scenarios:
            try:
                if streaming:
                    ds = self._open_monthlies_streaming(
                        run_dir,
                        keep_vars=keep_vars,
                        drop_vars=drop_vars,
                        chunks=chunks,
                        decode_times=decode_times,
                    )
                else:
                    ds = self._open_monthlies(run_dir)
                    if keep_vars is not None:
                        present = [v for v in keep_vars if v in ds.data_vars]
                        to_drop = [v for v in ds.data_vars if v not in present]
                        if to_drop:
                            ds = ds.drop_vars(to_drop)
                    elif drop_vars is not None:
                        to_drop = [v for v in drop_vars if v in ds.data_vars]
                        if to_drop:
                            ds = ds.drop_vars(to_drop)

            except (RuntimeError, FileNotFoundError, OSError, MemoryError) as e:
                warnings.warn(f"[FAUBIEnsemble] Skipping run {run_dir} ({lc},{pol}): {repr(e)}")
                skipped_runs.append((run_dir, lc, pol, repr(e)))
                continue

            # Tag scenario dims (lazy)
            ds = ds.expand_dims({"livestock": [lc], "country": [pol]})
            datasets.append(ds)

        if not datasets:
            raise RuntimeError(
                "No usable runs were loaded (all missing/corrupt or skipped). "
                "Relax strict_labels, adjust filename_glob, or call debug_report()."
            )

        combo = xr.combine_by_coords(
            datasets,
            combine_attrs="override",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        )
        # Enforce axis ordering when we know the full intended set.
        try:
            if self.countries:
                combo = combo.reindex(country=self.countries)
        except Exception:
            pass
        try:
            # Preserve livestock order from livestock_map keys
            if self.livestock_map:
                combo = combo.reindex(livestock=list(self.livestock_map.keys()))
        except Exception:
            pass


        # Friendly livestock_full labels where possible (fallback to the code as-is)
        try:
            combo = combo.assign_coords(
                livestock_full=("livestock", [
                    self.livestock_map.get(str(c), str(c)) for c in combo.livestock.values
                ])
            )
        except Exception:
            pass

        if skipped_runs:
            combo.attrs["faubi_skipped_runs"] = str(skipped_runs)

        self.ensemble_ds = combo
        return combo

    def load_reference(
        self,
        *,
        streaming: bool = True,
        keep_vars: Optional[List[str]] = None,
        drop_vars: Optional[List[str]] = None,
        chunks: Optional[Dict] = None,
        decode_times: bool = True,
    ) -> xr.Dataset:
        """
        Open the reference (no-animal) run for the year.
        Mirrors the ensemble loader options for consistency.
        """
        ref_dir = os.path.expanduser(self.reference_dir)
        if streaming:
            ds = self._open_monthlies_streaming(
                ref_dir,
                keep_vars=keep_vars,
                drop_vars=drop_vars,
                chunks=chunks,
                decode_times=decode_times,
            )
        else:
            ds = self._open_monthlies(ref_dir)
            if keep_vars is not None:
                present = [v for v in keep_vars if v in ds.data_vars]
                to_drop = [v for v in ds.data_vars if v not in present]
                if to_drop:
                    ds = ds.drop_vars(to_drop)
            elif drop_vars is not None:
                to_drop = [v for v in drop_vars if v in ds.data_vars]
                if to_drop:
                    ds = ds.drop_vars(to_drop)

        self.reference_ds = ds
        return ds

    def _open_statemet_streaming(
            self,
            run_dir: str,
            varname: str = "Met_BXHEIGHT",
            pattern: str = "GEOSChem.StateMet.*.nc4",
            chunks: dict | None = None,
            decode_times: bool = True,
    ) -> xr.Dataset:
        """
        Lazily open StateMet outputs (only `varname`) from a run dir's OutputDir,
        concat by time, and return a Dataset. Skips corrupt files and drops `ilev`.
        """
        files = sorted(glob.glob(str(Path(run_dir) / "OutputDir" / pattern)))
        if not files:
            raise FileNotFoundError(f"No StateMet files under {run_dir}/OutputDir matching: {pattern}")
        
        # Reuse the corruption filter
        files = self._filter_corrupt_files(files)
        if not files:
            raise RuntimeError(f"All StateMet files unreadable in {run_dir}")

        opened = []
        for f in files:
            try:
                ds = xr.open_dataset(
                    f,
                    engine="netcdf4",
                    decode_times=decode_times,
                    mask_and_scale=True,
                    chunks=chunks or {"time": 1},
                    cache=False,
                    lock=False,
                )
                # Drop ilev early if present
                if "ilev" in ds.dims:
                    ds = ds.drop_dims("ilev")
                elif "ilev" in ds.variables:
                    ds = ds.drop_vars("ilev")

                # Keep only the requested variable (plus coords)
                keep = [v for v in [varname] if v in ds.data_vars]
                if not keep:
                    warnings.warn(f"[FAUBIEnsemble] {varname} not found in {f}; skipping")
                    ds.close()
                    continue
                drop = [v for v in ds.data_vars if v not in keep]
                if drop:
                    ds = ds.drop_vars(drop)

                    opened.append(ds)
            except Exception as e:
                warnings.warn(f"[FAUBIEnsemble] Skipping StateMet file: {f}  <-- {repr(e)}")

            if not opened:
                raise RuntimeError(f"No usable StateMet files in {run_dir}")

        ds = xr.concat(
            opened,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )
        if "time" in ds:
            ds = ds.sortby("time")

        for src in opened:
            try: src.close()
            except Exception: pass

        return ds


    # ------------------------- Tile-based spot check -------------------------

    def spot_check_variability(
        self,
        var: str = "PM25",
        n_times: int = 3,
        n_tiles: int = 4,
        tile_h: int = 64,
        tile_w: int = 64,
        seed: int = 0,
        lev0: bool = True,
        max_bytes: int = 512 * 1024 * 1024,  # safety cap per tile compute
    ) -> dict:
        """
        Cheap variability check that only loads tiny spatial tiles.

        Strategy:
          1) Rechunk to small lat/lon tiles to make tiny reads cheap.
          2) Sample n_times random time indices.
          3) For each time, sample n_tiles random lat/lon tiles (tile_h x tile_w).
          4) For each tile, spatial-mean per scenario (livestock, country),
             then compute std across scenarios. Aggregate stats.

        Returns:
          {"mean_std_across_scenarios": ..., "max_std_across_scenarios": ...,
           "samples": <#tiles x times>, "tile_shape": (tile_h, tile_w)}
        """
        if self.ensemble_ds is None:
            raise RuntimeError("Call load_ensemble() first.")
        if var not in self.ensemble_ds:
            raise KeyError(f"{var} not found in ensemble dataset")

        da = self.ensemble_ds[var]
        if lev0 and "lev" in da.dims:
            da = da.isel(lev=0)

        for d in ("livestock", "country"):
            if d not in da.dims:
                raise ValueError(f"Missing dim '{d}' on {var}. Did you load the ensemble?")

        sz = da.sizes
        if "time" not in sz or "lat" not in sz or "lon" not in sz:
            raise ValueError("Expected dims to include 'time', 'lat', and 'lon'.")

        n_time = sz["time"]
        n_lat  = sz["lat"]
        n_lon  = sz["lon"]
        n_liv  = sz["livestock"]
        n_pol  = sz["country"]

        # Rechunk to small spatial tiles; keep time small; keep scenario dims whole
        da_tiled = da.chunk({
            "time": 1,
            "lat": min(tile_h, max(1, n_lat)),
            "lon": min(tile_w, max(1, n_lon)),
            "livestock": n_liv,
            "country": n_pol,
        })

        rng = np.random.default_rng(seed)

        def rand_start(n, tile):
            if n <= tile:
                return 0
            return int(rng.integers(0, n - tile))

        stats: List[float] = []

        # approximate per-tile memory
        dtype_bytes = 4 if (hasattr(da_tiled.data, "dtype") and str(da_tiled.dtype).startswith("float32")) else 8
        approx_tile_bytes = tile_h * tile_w * n_liv * n_pol * dtype_bytes
        if approx_tile_bytes > max_bytes:
            raise MemoryError(
                f"Tile read would exceed max_bytes: ~{approx_tile_bytes/1e6:.1f}MB > {max_bytes/1e6:.1f}MB. "
                f"Reduce tile_h/tile_w or raise max_bytes."
            )

        for _ in range(min(n_times, n_time)):
            t = int(rng.integers(0, n_time))
            for _ in range(n_tiles):
                i0 = rand_start(n_lat, tile_h)
                j0 = rand_start(n_lon, tile_w)
                tile = da_tiled.isel(
                    time=t,
                    lat=slice(i0, i0 + min(tile_h, n_lat - i0)),
                    lon=slice(j0, j0 + min(tile_w, n_lon - j0)),
                )
                # spatial mean for each scenario -> dims: (livestock, country)
                tile_mean = tile.mean(dim=("lat", "lon"), skipna=True)
                arr = tile_mean.compute().values  # tiny array (n_liv, n_pol)
                stats.append(float(np.nanstd(arr)))

        if not stats:
            raise RuntimeError("No stats collected; check dataset dimensions.")

        result = {
            "mean_std_across_scenarios": float(np.mean(stats)),
            "max_std_across_scenarios":  float(np.max(stats)),
            "samples": len(stats),
            "tile_shape": (tile_h, tile_w),
        }

        if result["max_std_across_scenarios"] == 0.0:
            raise AssertionError(
                f"Spot-check found zero variability across scenarios for '{var}'. "
                "Either runs are identical or path/labels are off."
            )
        return result

    # ------------------------- Diffs vs reference -------------------------

    def compute_pm25_diffs(self, var: str = "PM25", lev0: bool = True) -> xr.DataArray:
        """
        Return PM25 differences (scenario - reference) at lev=0 for all months/scenarios.
        Output dims: time, lat, lon, livestock, country
        """
        if self.ensemble_ds is None:
            raise RuntimeError("Call load_ensemble() first.")
        if self.reference_ds is None:
            raise RuntimeError("Call load_reference() first.")
        if var not in self.ensemble_ds or var not in self.reference_ds:
            raise KeyError(f"{var} must be present in both ensemble and reference datasets")

        scen = self.ensemble_ds[var]
        ref = self.reference_ds[var]
        if lev0 and "lev" in scen.dims:
            scen = scen.isel(lev=0)
        if lev0 and "lev" in ref.dims:
            ref = ref.isel(lev=0)

        # Broadcast reference across scenario dims
        ref_b = ref
        for d in ("livestock", "country"):
            if d not in scen.dims:
                raise ValueError(f"Ensemble missing '{d}' dimension on {var}.")
            if d not in ref_b.dims:
                ref_b = ref_b.expand_dims({d: scen[d]})

        scen, ref_b = xr.align(scen, ref_b, join="inner")
        diff = scen - ref_b
        diff.name = f"{var}_delta_vs_ref"
        diff.attrs["long_name"] = f"{var} (lev=0) scenario minus reference"
        self.diff_da = diff
        return diff

    def compute_pm25_diffs_AllOn(self, var: str = "PM25", lev0: bool = True) -> xr.DataArray:
        """
        Return PM25 differences for the AllOn framework (reference - scenario) at lev=0 for all months/scenarios.
        Output dims: time, lat, lon, livestock, country
        """
        if self.ensemble_ds is None:
            raise RuntimeError("Call load_ensemble() first.")
        if self.reference_ds is None:
            raise RuntimeError("Call load_reference() first.")
        if var not in self.ensemble_ds or var not in self.reference_ds:
            raise KeyError(f"{var} must be present in both ensemble and reference datasets")

        scen = self.ensemble_ds[var]
        ref = self.reference_ds[var]
        if lev0 and "lev" in scen.dims:
            scen = scen.isel(lev=0)
        if lev0 and "lev" in ref.dims:
            ref = ref.isel(lev=0)

        # Broadcast reference across scenario dims
        ref_b = ref
        for d in ("livestock", "country"):
            if d not in scen.dims:
                raise ValueError(f"Ensemble missing '{d}' dimension on {var}.")
            if d not in ref_b.dims:
                ref_b = ref_b.expand_dims({d: scen[d]})

        scen, ref_b = xr.align(scen, ref_b, join="inner")
        diff = ref_b - scen
        diff.name = f"{var}_delta_vs_ref"
        diff.attrs["long_name"] = f"{var} (lev=0) reference - scenario"
        self.diff_da = diff
        return diff

    # ------------------------- GIFs -------------------------

    def _month_label(self, val) -> str:
        try:
            return np.datetime_as_string(val, unit="D")[:7]
        except Exception:
            return str(val)

    def save_gif_for_scenario(
        self,
        livestock: str,
        country: str,
        out_dir: str = "gifs",
        fps: int = 2,
        use_cartopy: bool = False,
        force_vmax: Optional[float] = None,
        varname: str = "PM25_delta_vs_ref",
    ) -> str:
        """
        Save a GIF for a single (livestock, country) showing monthly PM25 deltas.
        Color scale is symmetric around zero (95th pct of |Δ| or force_vmax).
        """
        if self.diff_da is None:
            raise RuntimeError("Call compute_pm25_diffs() first.")

        da = self.diff_da
        if da.name != varname:
            pass  # allow renamed

        subset = da.sel(livestock=livestock, country=country)
        if subset.ndim != 3 or not all(d in subset.dims for d in ("time", "lat", "lon")):
            raise ValueError("Expected dims (time, lat, lon) after selecting scenario.")

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        outfile = Path(out_dir) / f"PM25_delta_{livestock}_{country}.gif"

        # Symmetric color scale
        if force_vmax is None:
            #vmax = float(np.nanpercentile(np.abs(subset.values), 99.99))
            vmax = float(np.nanmax(subset.values))
            if vmax == 0 or np.isnan(vmax):
                vmax = 1e-9
        else:
            vmax = float(force_vmax)
        vmin, vmax = -vmax, vmax
        #vmin, vmax = 0, vmax 

        frames = []
        months = subset["time"].values if "time" in subset.coords else range(subset.sizes["time"])

        if use_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

        for i in range(subset.sizes["time"]):
            arr = subset.isel(time=i).values

            if use_cartopy:
                fig = plt.figure(figsize=(9, 4.5))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                im = ax.imshow(
                    arr,
                    origin="lower",
                    extent=[float(subset.lon.min()), float(subset.lon.max()),
                            float(subset.lat.min()), float(subset.lat.max())],
                    vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                )
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
                ax.set_xlabel("lon index")
                ax.set_ylabel("lat index")

            title = (
                f"PM25 Δ vs Ref | Livestock={livestock}  "
                f"Country={country}  Month={self._month_label(months[i])}"
            )
            ax.set_title(title)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("μg m$^{-3}$ (approx)")

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        iio.imwrite(str(outfile), frames, loop=0, duration=1.0 / max(fps, 1))
        return str(outfile)

    def save_all_gifs(
        self,
        out_dir: str = "gifs",
        fps: int = 2,
        use_cartopy: bool = False,
        force_vmax: Optional[float] = None,
    ) -> List[str]:
        if self.diff_da is None:
            raise RuntimeError("Call compute_pm25_diffs() first.")
        paths = []
        for lc in self.diff_da.livestock.values:
            for pol in self.diff_da.country.values:
                p = self.save_gif_for_scenario(
                    livestock=str(lc),
                    country=str(pol),
                    out_dir=out_dir,
                    fps=fps,
                    use_cartopy=use_cartopy,
                    force_vmax=force_vmax,
                )
                paths.append(p)
        return paths

    # ------------------------- Diagnostics & Convenience -------------------------

    def debug_report(self) -> str:
        """
        Summarize what's on disk and any skipped files discovered so far.
        """
        base = Path(os.path.expanduser(self.ensemble_base_dir))
        pattern = self.filename_glob.format(year=self.year)

        lines = []
        lines.append(f"[FAUBIEnsemble] Base: {base}")
        lines.append(f"[FAUBIEnsemble] Pattern searched: **/OutputDir/{pattern}")

        output_dirs = [p for p in base.rglob("OutputDir") if p.is_dir()]
        lines.append(f"[FAUBIEnsemble] Found OutputDir folders: {len(output_dirs)}")

        zero_file_dirs = []
        for od in sorted(output_dirs):
            files = sorted(glob.glob(str(od / pattern)))
            lines.append(f"  - {od}: {len(files)} match(es)")
            if not files:
                zero_file_dirs.append(str(od))

        if zero_file_dirs:
            warnings.warn(
                "[FAUBIEnsemble] Some OutputDir folders have 0 matching files. "
                "Check filename_glob/year and actual filenames."
            )

        if self.skipped_files:
            lines.append("[FAUBIEnsemble] Skipped (corrupt/unreadable) files:")
            for run_dir, bad_list in self.skipped_files.items():
                lines.append(f"  - {run_dir}: {len(bad_list)} file(s)")
                for p in bad_list[:5]:
                    lines.append(f"      {p}")
                if len(bad_list) > 5:
                    lines.append(f"      ... (+{len(bad_list)-5} more)")

        return "\n".join(lines)

    def summary(self) -> str:
        parts = []
        if self._scenarios:
            parts.append(f"Scenarios discovered: {len(self._scenarios)}")
        if self.ensemble_ds is not None:
            parts.append(f"Ensemble shape: { {k: int(v) for k, v in self.ensemble_ds.sizes.items()} }")
        if self.reference_ds is not None:
            parts.append("Reference loaded ✓")
        if self.diff_da is not None:
            parts.append("PM25 deltas computed ✓")
        return " | ".join(parts) if parts else "No data loaded yet."

    def run_all(
        self,
        *,
        pm25_var: str = "PM25",
        # spot-check params (tile-based)
        n_times: int = 3,
        n_tiles: int = 4,
        tile_h: int = 64,
        tile_w: int = 64,
        seed: int = 0,
        # GIFs
        gif_out_dir: str = "gifs",
        use_cartopy: bool = False,
        fps: int = 2,
        force_vmax: Optional[float] = None,
        # loader options
        streaming: bool = True,
        keep_vars: Optional[List[str]] = None,
        drop_vars: Optional[List[str]] = None,
        chunks: Optional[Dict] = None,
        decode_times: bool = True,
        strict_labels: bool = False,
    ) -> Dict[str, object]:
        """
        End-to-end pipeline:
          - Discover & load ensemble (robust/streaming)
          - Tile-based spot-check variability
          - Load reference
          - Compute PM25 deltas
          - Write GIFs for all scenarios
        """
        self.load_ensemble(
            streaming=streaming,
            keep_vars=keep_vars,
            drop_vars=drop_vars,
            chunks=chunks,
            decode_times=decode_times,
            strict_labels=strict_labels,
        )
        checks = self.spot_check_variability(
            var=pm25_var,
            n_times=n_times,
            n_tiles=n_tiles,
            tile_h=tile_h,
            tile_w=tile_w,
            seed=seed,
        )
        self.load_reference(
            streaming=streaming,
            keep_vars=[pm25_var] if keep_vars is None else list(set(keep_vars + [pm25_var])),
            drop_vars=drop_vars,
            chunks=chunks,
            decode_times=decode_times,
        )
        self.compute_pm25_diffs(var=pm25_var, lev0=True)
        gif_paths = self.save_all_gifs(
            out_dir=gif_out_dir, fps=fps, use_cartopy=use_cartopy, force_vmax=force_vmax
        )
        return {"spot_checks": checks, "gif_paths": gif_paths}

    def compute_total_pm(
            self,
            *,
            which_diff: str = "diff_da",
            var: str = "PM25",
            statemet_pattern: str = "GEOSChem.StateMet.*.nc4",
            bxheight_var: str = "Met_BXHEIGHT",
            ug_per_tonne: float = 1e12,
            lev0: bool = True,
            sum_over_space: bool = False,
            decode_times: bool = True,
            chunks: dict | None = None,
    ) -> xr.DataArray:
        """
        Convert ΔPM2.5 (ug m^-3) to tonnes by multiplying:
        which_diff [ug m^-3] * Met_BXHEIGHT [m] * AREA(isel(lev=0)) [m^2] / 1e12 ug/tonne
        
        which_diff : str
        Name of the attribute on `self` holding the diff DataArray
        (e.g., "diff_da", "simple_scaling_diff", etc.).

        Returns:
        - If sum_over_space=False: DataArray with dims (time, lat, lon, livestock, country)
        - If sum_over_space=True : DataArray with dims (time, livestock, country) (spatial sum)
        """
        # --- resolve which diff DataArray to use ---
        if not hasattr(self, which_diff):
            raise RuntimeError(
                f"`self.{which_diff}` does not exist. "
                f"Make sure you have computed and stored the diff as `self.{which_diff}` "
                f"before calling compute_total_pm()."
            )

        diff_da = getattr(self, which_diff)
        
        if diff_da is None:
            raise RuntimeError(f"Call compute_pm25_diffs() first so `{which_diff}` exists.")
        if self.ensemble_ds is None:
            raise RuntimeError("Call load_ensemble() first so `AREA` is available.")

        if "AREA" not in self.ensemble_ds:
            raise KeyError("Ensemble dataset is missing 'AREA' variable needed for mass conversion.")

        # 1) Read StateMet (Met_BXHEIGHT) from *reference* directory
        ref_dir = os.path.expanduser(self.reference_dir)
        ds_sm = self._open_statemet_streaming(
            ref_dir,
            varname=bxheight_var,
            pattern=statemet_pattern,
            chunks=chunks,
            decode_times=decode_times,
        )
        if bxheight_var not in ds_sm:
            raise KeyError(f"'{bxheight_var}' not present in StateMet files.")

        bxh = ds_sm[bxheight_var]
        if lev0 and "lev" in bxh.dims:
            bxh = bxh.isel(lev=0)

        # 2) AREA at lev=0 (or broadcast if no lev)
        area = self.ensemble_ds["AREA"]
        if lev0 and "lev" in area.dims:
            area = area.isel(lev=0)

        # 3) Align and broadcast to scenario dims
        #    diff_da dims: (time, lat, lon, livestock, country)
        da = diff_da
        bxh, area, da = xr.align(bxh, area, da, join="inner")

        # Expand bxh/area to carry scenario dims
        for d in ("livestock", "country"):
            if d in da.dims and d not in bxh.dims:
                bxh = bxh.expand_dims({d: da[d]})
            if d in da.dims and d not in area.dims:
                area = area.expand_dims({d: da[d]})

        # 4) Compute tonnes
        total_pm = (da * bxh * area) / ug_per_tonne
        total_pm.name = f"{var}_delta_tonnes"
        total_pm.attrs.update({
            "long_name": f"{var} delta mass per grid cell",
            "units": "tonnes",
            "notes": f"{var} (ug/m^3) * Met_BXHEIGHT (m) * AREA (m^2) / {ug_per_tonne:g} ug/tonne",
        })

        # 5) Optional spatial sum -> (time, livestock, country)
        if sum_over_space:
            dims_to_sum = [d for d in ("lat", "lon") if d in total_pm.dims]
            total_pm = total_pm.sum(dim=dims_to_sum, skipna=True)
            total_pm.attrs["long_name"] = f"{var} delta mass (spatial sum)"

        # stash for reuse
        self.total_pm_da = total_pm
        return total_pm
