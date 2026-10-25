import importlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from A4.garment_recognition_project.benchmarking.benchmark_assignment3 import benchmark_assignment3
from A4.garment_recognition_project.evaluate_all import evaluate, DATASETS

mod = importlib.import_module(sys.argv[1])
cd = mod.train_classifier()
print('module', sys.argv[1])
here = os.path.dirname(os.path.realpath(__file__))
for ds in DATASETS:
    d = os.path.join(here, 'datasets', ds)
    hit = benchmark_assignment3(mod.im2segment, mod.segment2feature, mod.feature2class, cd, d, 0)[0]
    r = evaluate(mod.im2segment, mod.segment2feature, mod.feature2class, cd, d)
    print(f"{ds:7s} benchmark script: {hit*100:5.1f}% | all 5 garments: {r['hit_all5']*100:5.1f}% | "
          f"perfect segments: {r['hit_gt_segments']*100:5.1f}% | mean Jaccard {r['mean_jaccard']:.3f} | "
          f"images with right #segments {r['images_with_right_count']*100:5.1f}%", flush=True)
