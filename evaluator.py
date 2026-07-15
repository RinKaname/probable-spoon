import numpy as np

def lcs_length(a, b):
    # a and b are lists of strings
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def manhattan_dist(cell1, cell2):
    def parse_cell(c):
        parts = c.strip("c").split("_")
        return int(parts[0]), int(parts[1])
    x1, y1 = parse_cell(cell1)
    x2, y2 = parse_cell(cell2)
    return abs(x1 - x2) + abs(y1 - y2)

def calculate_row_score(pred_cells, true_cells):
    P = pred_cells
    T = true_cells
    len_P = len(P)
    len_T = len(T)

    # 1. set_f1
    set_P = set(P)
    set_T = set(T)
    if len_P == 0 and len_T == 0:
        set_f1 = 1.0
    elif len(set_P) == 0 or len(set_T) == 0:
        set_f1 = 0.0
    else:
        intersection = len(set_P.intersection(set_T))
        set_f1 = 2 * intersection / (len(set_P) + len(set_T))

    # 2. ordered_lcs
    if len_T == 0:
        ordered_lcs = 1.0 if len_P == 0 else 0.0
    else:
        ordered_lcs = lcs_length(P, T) / max(1, len_T)

    # 3. length_score
    if len_T == 0:
        length_score = 1.0 if len_P == 0 else 0.0
    else:
        length_score = np.exp(-abs(len_P - len_T) / max(2, len_T / 2))

    # 4. endpoint_score
    if len_P == 0 or len_T == 0:
        if len_P == 0 and len_T == 0:
            endpoint_score = 1.0
        else:
            endpoint_score = 0.0
    else:
        first_dist = manhattan_dist(P[0], T[0])
        last_dist = manhattan_dist(P[-1], T[-1])
        endpoint_score = (np.exp(-first_dist / 3) + np.exp(-last_dist / 3)) / 2

    # row_score
    row_score = 0.42 * set_f1 + 0.34 * ordered_lcs + 0.12 * length_score + 0.12 * endpoint_score
    return row_score

if __name__ == '__main__':
    # Test cases
    P = ["c07_13", "c08_13"]
    T = ["c07_13", "c08_13"]
    print("Perfect score:", calculate_row_score(P, T))

    P = ["c07_13"]
    print("Partial score:", calculate_row_score(P, T))
