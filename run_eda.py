import pandas as pd
import json
import numpy as np

def parse_cell(cell_str):
    parts = cell_str.strip("c").split("_")
    return int(parts[0]), int(parts[1])

train_df = pd.read_csv('train.csv')

stats = {
    'dist_vs_missing': [],
    'prefix_speed': [],
    'suffix_speed': [],
    'curve_deviation': []
}

print("Running EDA on 3328 rows...")

for idx, row in train_df.iterrows():
    sketch_data = json.loads(row['sketch_json'])
    true_data = json.loads(row['answer_json'])
    true_cells = true_data.get('hidden_cells', [])
    missing_count = row['missing_count_hint']

    prefix = []
    suffix = []

    for stroke in sketch_data.get('visible_strokes', []):
        if 'prefix' in stroke and 'suffix' in stroke and stroke.get('prefix') and stroke.get('suffix'):
            prefix = stroke['prefix']
            suffix = stroke['suffix']
            break

    if not prefix or not suffix or missing_count <= 0 or len(true_cells) == 0:
        continue

    prefix_pts = [np.array(parse_cell(p), dtype=float) for p in prefix]
    suffix_pts = [np.array(parse_cell(p), dtype=float) for p in suffix]
    true_pts = [np.array(parse_cell(p), dtype=float) for p in true_cells]

    start_coord = prefix_pts[-1]
    end_coord = suffix_pts[0]

    dist = np.linalg.norm(end_coord - start_coord)
    stats['dist_vs_missing'].append((dist, missing_count))

    if len(prefix_pts) > 1:
        n = min(3, len(prefix_pts))
        speed_sum = 0
        for i in range(1, n):
            speed_sum += np.linalg.norm(prefix_pts[-i] - prefix_pts[-i-1])
        avg_prefix_speed = speed_sum / (n - 1)
        stats['prefix_speed'].append(avg_prefix_speed)

    if len(suffix_pts) > 1:
        n = min(3, len(suffix_pts))
        speed_sum = 0
        for i in range(1, n):
            speed_sum += np.linalg.norm(suffix_pts[i] - suffix_pts[i-1])
        avg_suffix_speed = speed_sum / (n - 1)
        stats['suffix_speed'].append(avg_suffix_speed)

    if dist > 0:
        dir_vec = (end_coord - start_coord) / dist
        max_dev = 0
        for pt in true_pts:
            v = pt - start_coord
            proj = np.dot(v, dir_vec)
            proj = max(0, min(dist, proj))
            closest_pt = start_coord + proj * dir_vec
            dev = np.linalg.norm(pt - closest_pt)
            max_dev = max(max_dev, dev)
        stats['curve_deviation'].append((dist, missing_count, max_dev))

dist_missing_ratio = [m / max(1, d) for d, m in stats['dist_vs_missing']]
print(f"\n--- EDA RESULTS ---")
print(f"Average Missing Count / Distance Ratio: {np.mean(dist_missing_ratio):.2f}")
print(f"Max Missing Count / Distance Ratio (Complex shapes): {np.max(dist_missing_ratio):.2f}")
print(f"Percentage of highly complex shapes (Ratio > 2.0): {np.mean(np.array(dist_missing_ratio) > 2.0)*100:.2f}%")

prefix_speeds = stats['prefix_speed']
print(f"\nAverage Prefix Speed: {np.mean(prefix_speeds):.2f} units/point")
print(f"Max Prefix Speed: {np.max(prefix_speeds):.2f} units/point")
