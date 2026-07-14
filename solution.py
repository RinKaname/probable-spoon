import pandas as pd
import json
import numpy as np
from collections import defaultdict
import math

# ==========================================
# 1. Core Coordinate Parsers
# ==========================================
def parse_cell(cell_str):
    """Translates discrete string identifiers into Cartesian tuples."""
    parts = cell_str.strip("c").split("_")
    return int(parts[0]), int(parts[1])

def format_cell(x, y):
    """Projects Cartesian coordinates back into standard string format."""
    return f"c{x:02d}_{y:02d}"

# ==========================================
# 2. 2nd-Order Markovian State Transition
# ==========================================
def build_transition_model(train_df, alpha=0.01):
    """
    Constructs a 2nd-order Markov transition matrix tracking stroke curvature.
    Applies Laplace smoothing (alpha) to maintain absolute trellis continuity.
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    all_observed_moves = set()
    
    for _, row in train_df.iterrows():
        answer = json.loads(row['answer_json'])
        cells = answer.get('hidden_cells', [])
        
        if len(cells) < 4: # Need at least 4 cells to get 3 moves (2nd-order transition)
            continue
            
        coords = [parse_cell(c) for c in cells]
        
        moves = []
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            moves.append((dx, dy))
            all_observed_moves.add((dx, dy))
            
        # Track 2nd-order transitions: (move_{t-2}, move_{t-1}) -> move_t
        for i in range(2, len(moves)):
            prev_move = moves[i-2]
            curr_move = moves[i-1]
            next_move = moves[i]
            transition_counts[(prev_move, curr_move)][next_move] += 1
            
    transition_probs = defaultdict(dict)
    vocab_size = len(all_observed_moves)
    
    # Calculate probabilities with additive smoothing
    for state_tuple in list(transition_counts.keys()) + [m for m in all_observed_moves]: 
        # state_tuple can be a tuple of two moves, or we handle fallback 1st-order
        total_transitions = sum(transition_counts[state_tuple].values()) if state_tuple in transition_counts else 0
        
        for next_move in all_observed_moves:
            raw_count = transition_counts[state_tuple][next_move] if state_tuple in transition_counts else 0
            prob = (raw_count + alpha) / (total_transitions + alpha * vocab_size)
            transition_probs[state_tuple][next_move] = math.log(prob)
            
    return transition_probs, all_observed_moves

# ==========================================
# 3. A* Viterbi Dynamic Programming
# ==========================================
def fallback_interpolation(start_coord, end_coord, missing_count):
    """Parametric linear fallback for mathematically impossible trellis states."""
    x_vals = np.linspace(start_coord[0], end_coord[0], missing_count + 2)[1:-1]
    y_vals = np.linspace(start_coord[1], end_coord[1], missing_count + 2)[1:-1]
    
    x_vals = np.clip(np.round(x_vals), 0, 31).astype(int)
    y_vals = np.clip(np.round(y_vals), 0, 31).astype(int)
    
    return [format_cell(x, y) for x, y in zip(x_vals, y_vals)]

def predict_viterbi_path(sketch_json_str, missing_count, transition_probs, all_moves):
    """Executes a constrained dynamic programming trellis search with spatial heuristics."""
    sketch_data = json.loads(sketch_json_str)
    
    start_coord = None
    end_coord = None
    prev_momentum = (0, 0)
    init_momentum = (1, 0) 
    
    for stroke in sketch_data.get('visible_strokes', []):
        if 'prefix' in stroke and 'suffix' in stroke and stroke.get('prefix') and stroke.get('suffix'):
            prefix = stroke['prefix']
            suffix = stroke['suffix']
            
            start_coord = parse_cell(prefix[-1])
            end_coord = parse_cell(suffix[0])
            
            # Extract 2nd-order incoming inertia if available
            if len(prefix) >= 2:
                prev_coord = parse_cell(prefix[-2])
                init_momentum = (start_coord[0] - prev_coord[0], start_coord[1] - prev_coord[1])
            if len(prefix) >= 3:
                prev_prev_coord = parse_cell(prefix[-3])
                prev_momentum = (prev_coord[0] - prev_prev_coord[0], prev_coord[1] - prev_prev_coord[1])
            break
            
    if start_coord is None or end_coord is None:
        return json.dumps({"hidden_cells": []})
        
    if init_momentum not in all_moves:
        init_momentum = list(all_moves)[0] if all_moves else (1, 0)
    if prev_momentum not in all_moves:
        prev_momentum = init_momentum

    # dp[time_step][(current_x, current_y, prev_dx, prev_dy, curr_dx, curr_dy)] = (log_prob, prev_state)
    dp = {0: {(start_coord[0], start_coord[1], prev_momentum[0], prev_momentum[1], init_momentum[0], init_momentum[1]): (0.0, None)}}
    
    heuristic_weight = 0.5 # Tune this to penalize drifting. Higher = straighter lines to target.

    for t in range(1, missing_count + 1):
        dp[t] = {}
        for (x, y, pdx, pdy, cdx, cdy), (log_prob, _) in dp[t-1].items():
            for ndx, ndy in all_moves:
                nx, ny = x + ndx, y + ndy
                
                if 0 <= nx < 32 and 0 <= ny < 32:
                    # Query 2nd-order probability
                    state_tuple = ((pdx, pdy), (cdx, cdy))
                    trans_prob = transition_probs.get(state_tuple, {}).get((ndx, ndy), -1e9)
                    
                    # A* Heuristic: Penalize distance to the final target based on remaining steps
                    dist_to_target = math.sqrt((end_coord[0] - nx)**2 + (end_coord[1] - ny)**2)
                    steps_remaining = missing_count - t
                    # Ideal distance is proportional to steps left. Drift incurs a penalty.
                    distance_penalty = abs(dist_to_target - steps_remaining) * heuristic_weight
                    
                    new_log_prob = log_prob + trans_prob - distance_penalty
                    
                    state_key = (nx, ny, cdx, cdy, ndx, ndy)
                    if state_key not in dp[t] or new_log_prob > dp[t][state_key][0]:
                        dp[t][state_key] = (new_log_prob, (x, y, pdx, pdy, cdx, cdy))
                        
    # Terminal Constraint Forcing
    best_final_state = None
    max_final_prob = -float('inf')
    
    if missing_count in dp and dp[missing_count]:
        for (x, y, pdx, pdy, cdx, cdy), (log_prob, _) in dp[missing_count].items():
            # Check final required jump to the absolute end_coord
            final_dx = end_coord[0] - x
            final_dy = end_coord[1] - y
            
            state_tuple = ((pdx, pdy), (cdx, cdy))
            final_trans_prob = transition_probs.get(state_tuple, {}).get((final_dx, final_dy), -1e9)
            
            total_prob = log_prob + final_trans_prob
            if total_prob > max_final_prob:
                max_final_prob = total_prob
                best_final_state = (x, y, pdx, pdy, cdx, cdy)
                
    if best_final_state is None or max_final_prob < -1e8:
        predicted_cells = fallback_interpolation(start_coord, end_coord, missing_count)
    else:
        predicted_cells = []
        curr_state = best_final_state
        for t in range(missing_count, 0, -1):
            predicted_cells.append(format_cell(curr_state[0], curr_state[1]))
            curr_state = dp[t][curr_state][1]
        predicted_cells.reverse()
        
    return json.dumps({"hidden_cells": predicted_cells})

# ==========================================
# 4. Pipeline Execution
# ==========================================
def main():
    train_path = './dataset/public/train.csv'
    test_path = './dataset/public/test.csv'
    output_path = './working/submission.csv'
    
    print("Initializing sequence environment...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Compiling 2nd-order Markov transitions...")
    transition_probs, all_moves = build_transition_model(train_df)
    
    print("Executing constrained A* pathfinding...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        pred_json = predict_viterbi_path(
            row['sketch_json'], 
            row['missing_count_hint'], 
            transition_probs,
            all_moves
        )
        predictions.append({'id': row['id'], 'answer_json': pred_json})
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} / {len(test_df)} sequences.")
            
    print(f"Serializing optimized arrays to {output_path}...")
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()