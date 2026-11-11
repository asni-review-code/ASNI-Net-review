# Figure provenance and attribution

`architecture.pdf`, `adaptive_sampling.pdf` and `node_interaction.pdf` show the current English manuscript's Figures 1–3. Document-level author
and other metadata have been removed for anonymous review; rendered content
has been verified pixel-for-pixel against the source figures. The corresponding PNGs
are raster renderings for GitHub display. No model structure or image content
was altered. The architecture figure's image and output are BUS-BRA test sample
`bus_0200-s` and the archived ASNI-Net / 20260928 prediction; that pair was
selected for illustration, not as a representative performance estimate.

`results.png` / `results.svg` are generated from the supplied three-seed JSON
by [tools/render_results.py](../tools/render_results.py). Dots are seeds and
error bars are sample SD. The charts contain no comparator results.

`qualitative_bus_bra.png` contains three BUS-BRA test examples selected nearest
the 10th, 50th and 90th percentiles of archived per-image Dice (seed 20260928).
The exact IDs and pixel-count verification are in [examples.json](examples.json).
This is post-hoc illustration, with full-field images and no new inference,
threshold selection or model tuning. Plot assembly resizes panels for display;
all metrics refer to original image dimensions.

BUS-BRA images and annotations are credited to **Wilfrido Gómez-Flores,
Maria Julia Gregorio-Calas, and Wagner Coelho de Albuquerque Pereira**,
“BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis
Systems,” *Medical Physics* 51, 3110–3123 (2024),
[doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Data source: [Zenodo v1.0](https://zenodo.org/records/8231412).
The original dataset copyright/permission text is retained in
[bus_bra_dataset_license.txt](bus_bra_dataset_license.txt). This data notice is
separate from the software MIT license. No full dataset is redistributed.
