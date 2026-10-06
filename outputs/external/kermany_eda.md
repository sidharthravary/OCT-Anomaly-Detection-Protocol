# Kermany OCT test set: EDA (vs NEH)

- Images: 1000; per class: {'CNV': 250, 'DME': 250, 'DRUSEN': 250, 'NORMAL': 250}
- Patients: 635 (images per patient: median 1, max 18)
- Test patients that also appear in Kermany *train*: 0 (0%)
- File format: .jpeg for all; PIL modes {'L': 998, 'RGB': 2}; identical channels / single channel: 100%
- Image sizes (width x height): 642x277: 1, 768x496: 4, 900x455: 1, 1024x496: 942, 1536x496: 52
- Near-white pixels (>= 250): mean 6.0%; images with > 5% near-white: 42% (NEH: black background, no white areas)
- Near-black pixels (<= 5): mean 12.3%

NEH for comparison: 768 x 496 for every image, TIF 79% / JPG 21%, greyscale stored as RGB, burned-in scale bar (masked), full volumes with per-slice labels.
