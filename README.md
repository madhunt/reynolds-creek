# Beamforming Analysis of Infrasound Signals from Rangeland Fire
Array processing code and results from infrasound data collected at the Reynolds Creek Experimental Watershed (RCEW) prescribed burn on 06-07 Oct 2023. 


## Repository Overview

`code`
    `plotting_code`
    `sbatch_scripts`
    `tests`

`figures`

For data, including raw miniseed files, GPS survey coordinates, and processed beamforming results, contact M. Hunt.

### Workflow
1. **Pre-Process Data**: Use `preprocess.py` to plot raw traces and inspect. Note if any traces should be removed from processing. Re-run to pre-process data (e.g. shift traces back by an integer number of seconds). This script will read raw mseed files from `data/raw` and save pre-processed data in `data/mseed` directory. 

2. **Process Data**: Use `beamform.py` to process data with conventional shift-and-stack beamforming. This script will produce backazimuth and slowness plots in `figures`, and will save processed data as a pkl file to `data/processed`. 



3. **Crossbeam**

4. **Create Figures**


## Field Site
The study region is the Reynolds Creek Experimental Watershed (RCEW), located in the Owyhee Mountains of southwestern Idaho, about 80 km southwest of Boise, ID. A prescribed burn was carried out in RCEW by the Bureau of Land Management on 06 Oct 2023, with an intended purpose of decreasing the juniper population for grazing cattle. The proposed prescribed burn area was approximately 9.4 km^2. The elevation of the burn region ranges from 1457 m to 1870 m, with slopes from 0-25%.

In total, 75 infrasound sensors were deployed in arrays within the burn area and surrounding region between Aug and Oct 2023, including the 44-element TOP array.

<figure>
<p align="center">
    <img src="figures/burn_severity_map.png" width="400">
    <figcaption> <i> Severity of RCEW prescribed burn on 06 Oct 2023. Normalized Burn Ratio (NBR) calculated with B8A and B12 (NIR, SWIR) from 20 m resolution Sentinel-2 images corrected to surface reflectance. Differenced NBR (dNBR), calculated between pre-burn (2023-09-28) and post-burn (2023-10-08) images, ranges from unburned land (light grey) to low severity (yellow) to moderate severity (dark red). Proposed extent of prescribed burn (black outline) sourced from the BLM. Infrasound sensor arrays (blue triangles) surveyed in Oct 2023. Elevation contour lines (dark grey) are 1:24000, sourced from the USGS. </i> </figcaption>
</p>
</figure>




## Beamforming Results




<figure>
<p align="center">
    <img src="figures/backaz_and_heli_2.0-8.0Hz_20231006-15-00_20231007-02-00.png" width="500">
    <figcaption> <i> Backazimuth over time for data filtered between 2-8 Hz, the fire infrasound band. </i> </figcaption>
</p>
</figure>




<figure>
<p align="center">
    <img src="figures/backaz_and_heli_24.0-32.0Hz20231007-16-00)20231007-21-00.png" width="500">
    <figcaption> <i> Backazimuth over time for data filtered between 24-32 Hz, the helicopter infrasound band. </i> </figcaption>
</p>
</figure>