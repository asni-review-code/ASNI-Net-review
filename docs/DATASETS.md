# Datasets and fixed membership

The manifests provide all image IDs, roles, group IDs, source image/mask hashes,
dimensions and pixel hashes needed to reconstruct the paper's exact inputs.
Raw data are obtained from the original providers, not from this repository.
The `original_*_path` columns are relative provenance hints from the research
archive, not requirements to recreate the author's directory tree. `prepare`
finds input files by content hash and writes portable `dataset/images` and
`dataset/masks` directories.

| Dataset | Version / official source | Train | Val | Test |
|---|---|---:|---:|---:|
| BUS-BRA | [Zenodo v1.0](https://zenodo.org/records/8231412), outer fold 1 | 1204 | 295 | 376 |
| TN3K | [Author dataset](https://huggingface.co/datasets/haifan-gong/TN3K), revision `c0479e921fc9b7afcf1e8ca9b296a408e06a5bb9` | 2591 | 288 | 614 |
| MMOTU / OTU_2d | [Author repository](https://github.com/cv516Buaa/MMOTU_DS2Net), [author files](https://drive.google.com/drive/folders/1c5n0fVKrM9-SZE1kacTXPt1pt844iAs1) | 703 | 82 | 382 |
| BUS-UCLM | [Mendeley Data v3](https://data.mendeley.com/datasets/7fvgj4jsp7/3) | — | — | 665 external |

Source access and original audit date: 2026-09-23. BUS-BRA and BUS-UCLM were
recorded as CC BY 4.0; TN3K's author dataset card declared MIT. An explicit
MMOTU dataset redistribution license was not established; the repository's
code license is not permission to redistribute its images. Obtain the data
from the providers and comply with their current terms and citation requirements.

Label conversion: BUS-BRA/MMOTU original binary mask >0; TN3K JPEG mask >=128;
BUS-UCLM red/green lesion pixels become 1 and black becomes 0. Unknown UCLM
colors fail validation. Output masks are 0/1 PNG. Pixel hashes allow equivalent
PNG encodings across Pillow versions; original byte hashes are also retained.

BUS-BRA grouping uses provided patient IDs. TN3K/MMOTU patient independence
cannot be independently verified from the available metadata. Similarity-based
screening is not proof of patient separation. Do not claim otherwise based on
the supplied group column.

MMOTU excluded 267 normal-ovary images and 35 near-duplicate author-training
images, giving 1167 lesion images. `splits/mmotu_curation.csv` records the full
selection and exclusions. The same-scene screen used dHash distance <=4 with
128×128 grayscale correlation >=0.90, or distance <=2 with correlation >=0.85.
No train/test membership is inferred anew by this release.

BUS-UCLM uses 665 eligible, unique, non-conflicting images from the 683-image
release: 258 lesion images and 407 normal images. It remains external-only;
neither checkpoints nor thresholds are selected on it. Use the fixed BUS-BRA
20260928 weights for the paper's single-checkpoint external evaluation.

If source hashes do not match, stop and check the version/extraction. Do not
silently regenerate split hashes or convert an incompatible dataset into a new
claim of reproducing the published result.
