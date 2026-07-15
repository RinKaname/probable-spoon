# Model Architecture

To strictly adhere to the CPU compute limitations while aggressively optimizing for the `ordered_lcs` and length-dependent sequence constraints, I circumvented over-parameterized neural networks and localized discrete state models entirely. The architecture relies on a **Parametric Geometric Curve Interpolation** framework, specifically utilizing highly-tuned **Cubic Bezier Curves** coupled with **Momentum-Weighted Tangent Estimation**. This geometric framework calculates localized stroke inertia from historical prefix arrays and mathematically smooths a continuous continuous space trajectory directly onto the discrete 32x32 target grid.

# Preprocessing

The localized data preparation was entirely geometric and sequence-oriented.
1.  **Cartesian Projection:** Discrete string identifiers (e.g., `"c07_13"`) were decoded into floating-point Cartesian tuples to allow continuous-space mathematical operations.
2.  **Stroke Gap Extraction:** The architecture dynamically parsed the `sketch_json` to extract the terminal prefix sequence and the origin suffix sequence, bounding the exact geometric problem space.
3.  **Inertial Weighting:** Rather than using discrete matrix transitions, the previous $N$ coordinates in the prefix and suffix were transformed into continuous velocity vectors (tangents), applying an exponentially decaying time-weight to heavily bias the immediate momentum of the pen stroke just prior to the gap.

# Key Design Decisions

1.  **Cubic Bezier Interpolation:** Rather than predicting discrete, unlinked cells sequentially, a single mathematical curve ($P(t)$) was defined using four control points: the exact start, the exact end, and two dynamically generated control points dictated by the calculated inertial tangents.
2.  **Dynamic Curve Amplitude Scaling:** A critical design pivot was mathematically estimating the "bend" required to bridge the gap. By calculating the dot product of the incoming and outgoing tangent vectors relative to the straight-line distance and the `missing_count_hint`, the algorithm dynamically amplifies the control points. This effectively pulls the curve outward to naturally mimic the looping curvature of human hand strokes when traversing large temporal gaps over short distances.
3.  **Arc-Length Uniform Resampling:** To satisfy the strict `length_score` constraints and avoid clustering near control points (a fundamental property of parametric $t$ evaluation in Bezier mathematics), the continuous trajectory was densely supersampled and integrated to calculate true arc length. The final discrete cells were extracted by sampling perfectly uniformly along this geometric arc.

# What Worked and What Didn't

**What Didn't Work:**
Attempting to utilize pure Catmull-Rom splines resulted in overly aggressive oscillations when the stroke gap required a sharp turn, failing to respect the user's intended trajectory. Furthermore, standard parametric sampling (evaluating $t$ evenly from 0.0 to 1.0) clustered grid predictions heavily at the start and end of the gap, destroying the sequential integrity required for `ordered_lcs` and causing consecutive cell duplicates which severely penalized the final scoring output.

**What Worked:**
Shifting to the geometrically bounded Cubic Bezier curve, combined with the arc-length uniform resampling algorithm, perfectly preserved the spatial continuity of the stroke. Introducing an exponential decay formula (e.g., $e^{-0.4 * i}$) to estimate the tangent vectors allowed the algorithm to "listen" to the last 4-5 points of the pen stroke without being heavily distorted by older, irrelevant geometry. Finally, implementing a dynamic scalar to automatically amplify the curve amplitude when a "U-Turn" was mathematically detected allowed the CPU-bound solution to score exceptionally high across all complex, non-linear gaps.