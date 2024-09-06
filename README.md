# Reynolds Creek Experimental Watershed
Array processing code and results from Reynolds Creek Experimental Watershed (RCEW) prescribed burn.


## Repository Overview

All data stored in `data` folder, including raw miniseed files, survey coordinates, and processed data.

### Workflow
1. **Pre-Process Data**: Use `preprocess.py` to plot raw traces and inspect. Note if any traces should be removed from processing. Re-run to pre-process data (e.g. shift traces back by an integer number of seconds). This script will read raw mseed files from `data/raw` and save pre-processed data in `data/mseed` directory. 

2. **Process Data**: Use `beamform.py` to process data with conventional shift-and-stack beamforming. This script will produce backazimuth and slowness plots in `figures`, and will save processed data as a pkl file to `data/processed`. 


## Field Site
The study region is the Reynolds Creek Experimental Watershed (RCEW), located in the Owyhee Mountains of southwestern Idaho, about 80 km southwest of Boise, ID. A prescribed burn was carried out in RCEW by the Bureau of Land Management on 06 Oct 2023, with an intended purpose of decreasing the juniper population for grazing cattle. The proposed prescribed burn area was approximately 9.4 km^2. The elevation of the burn region ranges from 1457 m to 1870 m, with slopes from 0-25%.

In total, 75 infrasound sensors were deployed in arrays within the burn area and surrounding region between Aug and Oct 2023, including the 44-element TOP array.


<figure>
<p align="center">
    <img src="/figures/Burn_Severity.png" width="400">
    <figcaption> Severity of RCEW prescribed burn on 06 Oct 2023. Normalized Burn Ratio (NBR) calculated with B8A and B12 (NIR, SWIR) from 20 m resolution Sentinel-2 images corrected to surface reflectance. Differenced NBR (dNBR), calculated between pre-burn (2023-09-28) and post-burn (2023-10-08) images, ranges from unburned land (light grey) to low severity (yellow) to moderate severity (dark red). Proposed extent of prescribed burn (black outline) sourced from the BLM. Infrasound sensor arrays (blue triangles) surveyed in Oct 2023. Elevation contour lines (dark grey) are 1:24000, sourced from the USGS. </figcaption>
</p>
</figure>


<figure>
<p align="center">
    <img src="/figures/Overview_Map.png" width="400">
    <figcaption> Overview map of study area in the Reynolds Creek Experimental Watershed area. False-color images with B8, B4, B3 (NIR, Red, Green) image, highlighting (a) vegetation in red for pre-burn image (2023-09-28), and (b) burned area in green in the post-burn image (2023-10-18). Proposed perimeter of burn region (blue outline) was sourced from the BLM. Infrasound sensors arrays (yellow triangles) were deployed and surveyed throughout Aug-Oct 2023. Images are 10 m resolution Sentinel-2 L2A products converted to surface reflectance. (c) Overview map, showing location of the site in southwest Idaho. Idaho boundary and counties sourced from ISU. </figcaption>
</p>
</figure>


