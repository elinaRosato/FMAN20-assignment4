# Image Analysis – Assignment 4

My solution to **Assignment 4** of the **Image Analysis** course (**FMAN20**) at **Lund University**. The write-up is in [`report/report.pdf`](report/report.pdf).

## The 4 tasks

1. **Color correction** (`task1/`) – estimates the 3×3 matrix that changes the colors of one image into another with RANSAC, corrects the image, and compares the result with PSNR, SSIM and FLIP.
2. **Segmentation with graph cuts** (`task2/`) – segments the two heart chambers in an MRI image with a Gaussian data term and max-flow/min-cut.
3. **Computer vision** – a pen-and-paper calculation (only in the report).
4. **Garment classification** (`task4/`) – the full system (segmentation, features, classifier) tested on five datasets. `version1.py` is the system from Assignment 3 (hitrates 30–60 %) and `assignment4.py` is the improved version 2 (74–82 %).

## Repository structure

- `report/` – the LaTeX source (`report.tex`), the figures it uses and the compiled `report.pdf`.
- `task1/`, `task2/` – the script of each task. Input and output images are in the `assets/` folder next to the script (`task2/` also has `heart_data.mat`).
- `task4/` – `assignment4.py` (final system), `version1.py`, the benchmark code in `benchmarking/`, the training data (`garmentsegmentdata.npy`, `garmentsegmentgt.npy`) and the five datasets in `datasets/`. `evaluate_all.py` and `run_eval.py` give the extra numbers for the segmentation analysis in the report.
- `inl4_2026.pdf` – the assignment description.

## Running the code

The scripts find their files from their own location, so they can be run from any folder. For example from the repository root:

```bash
pip install numpy scipy matplotlib scikit-image scikit-learn flip-evaluator

python3 task1/task1.py
python3 task2/task2.py
python3 task4/assignment4.py     # prints the hitrate on the five datasets
python3 task4/version1.py        # the same for version 1
python3 task4/run_eval.py assignment4   # extra analysis, use version1 for version 1
```

The report is built with `latexmk` (or `pdflatex`) from the `report/` folder:

```bash
cd report
latexmk -pdf report.tex
```
