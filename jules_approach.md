# Model Architecture

To strictly adhere to the CPU compute limitations while aggressively optimizing for the `ordered_lcs` and length-dependent sequence constraints, I circumvented over-parameterized neural networks and localized discrete state models entirely. The architecture relies on a **Parametric Geometric Curve Interpolation** framework, specifically utilizing highly-tuned **Cubic Bezier Curves** coupled with **Momentum-Weighted Tangent Estimation** and **Dynamic Amplitude Bending**. This geometric framework calculates localized stroke inertia from historical prefix arrays and mathematically smooths a continuous space trajectory directly onto the discrete 32x32 target grid.

# Preprocessing

The localized data preparation was entirely geometric and sequence-oriented.
1.  **Cartesian Projection:** Discrete string identifiers (e.g., `"c07_13"`) were decoded into floating-point Cartesian tuples to allow continuous-space mathematical operations.
2.  **Stroke Gap Extraction:** The architecture dynamically parsed the `sketch_json` to extract the terminal prefix sequence and the origin suffix sequence, bounding the exact geometric problem space.
3.  **Inertial Weighting:** Rather than using discrete matrix transitions, the previous $N$ coordinates (up to 5) in the prefix and suffix were transformed into continuous velocity vectors (tangents), applying an exponential decay ($e^{-0.4 * i}$) to heavily bias the immediate momentum of the pen stroke just prior to the gap.

# Key Design Decisions

1.  **Cubic Bezier Interpolation:** Rather than predicting discrete, unlinked cells sequentially, a single mathematical curve ($P(t)$) was defined using four control points: the exact start, the exact end, and two dynamically generated control points dictated by the calculated inertial tangents.
2.  **Dynamic Curve Amplitude Scaling & Bend Boosting:** An extensive EDA proved that the `missing_count_hint` divided by the straight-line geometric distance acts as a highly predictive ratio for the complexity of the hidden shape. If this ratio exceeded a structural threshold (1.1x), a massive geometric multiplier (1.5x) was applied to the curve control points.
3.  **Orthogonal Bend Factor Integration:** Furthermore, the dot product of the incoming and outgoing normalized tangents was utilized to map a specific "bend_factor" bonus (up to 0.2). If a stroke physically entered the gap pointing in the exact opposite direction it exited (mathematically indicating a complex 'U-turn' loop), the control point scalars were organically boosted.
4.  **Arc-Length Uniform Resampling:** To satisfy the strict `length_score` constraints and avoid clustering near control points (a fundamental property of parametric $t$ evaluation in Bezier mathematics), the continuous trajectory was densely supersampled ($N * 20$) and integrated to calculate true arc length. The final discrete cells were extracted by sampling perfectly uniformly along this geometric arc, maintaining identical index pacing across the gap.

# What Worked and What Didn't

**What Didn't Work:**
Attempting to utilize pure Catmull-Rom splines resulted in overly aggressive oscillations when the stroke gap required a sharp turn, frequently over-projecting the trajectory. Standard parametric sampling (evaluating $t$ evenly from 0.0 to 1.0) clustered grid predictions heavily at the start and end of the gap, destroying the sequential integrity required for `ordered_lcs` and causing consecutive cell duplicates which severely penalized the final scoring output. Attempting to explicitly filter out consecutive duplicate coordinates post-generation resulted in mismatched lengths and negatively affected the baseline.

**What Worked:**
Shifting to the geometrically bounded Cubic Bezier curve, combined with the arc-length uniform resampling algorithm, perfectly preserved the spatial continuity of the stroke. Combining the exponentially decayed velocity vectors with the heavily analyzed Bend Factor matrix pushed the score from the original ~0.59 to > 0.60 by correctly modeling severe looping motions in high-density areas that straight-line interpolations completely failed on.
