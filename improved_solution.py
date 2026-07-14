import json
import numpy as np

def parse_cell(cell_str):
    parts = cell_str.strip("c").split("_")
    return int(parts[0]), int(parts[1])

def format_cell(x, y):
    x = max(0, min(31, int(round(x))))
    y = max(0, min(31, int(round(y))))
    return f"c{x:02d}_{y:02d}"

def bezier_curve(p0, p1, p2, p3, t):
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def predict(sketch_json_str, missing_count):
    scale_miss = 0.34
    amp_ratio = 1.1
    amp_factor = 1.5
    n_pts = 5
    decay = 0.4

    sketch_data = json.loads(sketch_json_str)
    prefix = []
    suffix = []

    for stroke in sketch_data.get('visible_strokes', []):
        if 'prefix' in stroke and 'suffix' in stroke and stroke.get('prefix') and stroke.get('suffix'):
            prefix = stroke['prefix']
            suffix = stroke['suffix']
            break

    if not prefix or not suffix or missing_count <= 0:
        return json.dumps({"hidden_cells": []})

    prefix_pts = [np.array(parse_cell(p), dtype=float) for p in prefix]
    suffix_pts = [np.array(parse_cell(p), dtype=float) for p in suffix]

    start_coord = prefix_pts[-1]
    end_coord = suffix_pts[0]

    tangent_start = np.zeros(2)
    weight_sum = 0
    n = min(n_pts, len(prefix_pts))
    for i in range(1, n):
        w = np.exp(-decay * i)
        tangent_start += (prefix_pts[-i] - prefix_pts[-i-1]) * w
        weight_sum += w
    if weight_sum > 0: tangent_start /= weight_sum

    tangent_end = np.zeros(2)
    weight_sum = 0
    n = min(n_pts, len(suffix_pts))
    for i in range(1, n):
        w = np.exp(-decay * i)
        tangent_end += (suffix_pts[i] - suffix_pts[i-1]) * w
        weight_sum += w
    if weight_sum > 0: tangent_end /= weight_sum

    if np.linalg.norm(tangent_start) == 0:
        tangent_start = end_coord - start_coord
    if np.linalg.norm(tangent_end) == 0:
        tangent_end = end_coord - start_coord

    dist = np.linalg.norm(end_coord - start_coord)

    scale_start = missing_count * scale_miss
    scale_end = missing_count * scale_miss

    ratio = missing_count / (dist + 1e-5)

    norm_start = np.linalg.norm(tangent_start)
    norm_end = np.linalg.norm(tangent_end)

    bend_factor = 0.5
    if norm_start > 0 and norm_end > 0:
        dir_start = tangent_start / norm_start
        dir_end = -tangent_end / norm_end
        dot_prod = np.dot(dir_start, dir_end)
        bend_factor = (1 - dot_prod) / 2.0

    factor = 1.0
    if ratio > amp_ratio:
        factor = amp_factor + (bend_factor * 0.2)

    scale_start *= factor
    scale_end *= factor

    if norm_start > 0: tangent_start = tangent_start / norm_start * scale_start
    if norm_end > 0: tangent_end = tangent_end / norm_end * scale_end

    p0 = start_coord
    p1 = start_coord + tangent_start
    p2 = end_coord - tangent_end
    p3 = end_coord

    dense_t = np.linspace(0, 1, max(200, missing_count * 20))
    dense_points = [bezier_curve(p0, p1, p2, p3, t) for t in dense_t]

    lengths = [0.0]
    for i in range(1, len(dense_points)):
        l = np.linalg.norm(dense_points[i] - dense_points[i-1])
        lengths.append(lengths[-1] + l)

    total_length = lengths[-1]

    if total_length == 0:
        t_values = np.linspace(0, 1, missing_count + 2)[1:-1]
        exact_cells = [format_cell(*bezier_curve(p0, p1, p2, p3, t)) for t in t_values]
        return json.dumps({"hidden_cells": exact_cells})

    target_lengths = np.linspace(0, total_length, missing_count + 2)[1:-1]

    exact_cells = []
    for target in target_lengths:
        idx = np.searchsorted(lengths, target)
        if idx == 0: pt = dense_points[0]
        elif idx == len(lengths): pt = dense_points[-1]
        else:
            l0, l1 = lengths[idx-1], lengths[idx]
            p0_dense, p1_dense = dense_points[idx-1], dense_points[idx]
            f = (target - l0) / (l1 - l0) if l1 > l0 else 0
            pt = p0_dense + f * (p1_dense - p0_dense)

        exact_cells.append(format_cell(pt[0], pt[1]))

    return json.dumps({"hidden_cells": exact_cells})

def main():
    import pandas as pd
    test_path = 'test.csv'
    output_path = 'submission.csv'

    print("Generating submission...")
    test_df = pd.read_csv(test_path)

    predictions = []
    for idx, row in test_df.iterrows():
        pred_json = predict(row['sketch_json'], row['missing_count_hint'])
        predictions.append({'id': row['id'], 'answer_json': pred_json})

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} / {len(test_df)} sequences.")

    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print("Pipeline execution complete. Saved to submission.csv")

if __name__ == "__main__":
    main()
