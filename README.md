# A-Screening-Level-Multi-Index-Framework-for-Assessing-Surface-Water-Quality-Trends-
Implements a reproducible multi-index framework for assessing surface water quality trends using Theil–Sen slopes, Kendall's tau, inter-period median differences, bootstrap stability, permutation tests, and sensitivity analysis under data-scarce monitoring conditions.

Here is a polished scientific description suitable for a manuscript, supplementary material, or a GitHub repository:

---

## Code Description

This script implements a **reproducible multi-index framework for assessing surface water quality trends in the Atrato River Basin (Colombia)** under data-scarce monitoring conditions and artisanal gold mining pressure. The methodology follows the analytical workflow described in the manuscript and is designed to operate with short time series and heterogeneous water-quality indices.

The script accepts a CSV file containing annual observations for each monitoring station and water quality index (ICA, ICOMO, ICOMI, ICOSUS, ICOMINERIA, and ICOTRO) and produces a comprehensive set of trend metrics and classifications for each station–index combination.

The analytical workflow consists of the following steps:

1. **Index orientation normalization**
   All indices are transformed into a common orientation so that positive changes consistently represent improvements in water quality, regardless of the original index convention.

2. **Theil–Sen slope estimation**
   A robust estimate of the temporal trend is computed for each index using the median of pairwise slopes, allowing the detection of gradual improvements or deteriorations.

3. **Kendall's tau coefficient and significance testing**
   The monotonic association between index values and time is quantified using Kendall's tau, together with its corresponding p-value.

4. **Inter-period median difference analysis**
   Median index values are compared between two time periods (2020–2022 and 2023–2025 by default) to identify shifts in water quality conditions.

5. **Trend classification**
   Each station–index series is classified into one of five categories:

   * **Improving**
   * **Stable**
   * **Deteriorating**
   * **Mixed/Unclear**
   * **Insufficient data**

   The classification is based on proportional thresholds applied to both the Theil–Sen slope and the inter-period median difference.

6. **Bootstrap classification stability**
   A bootstrap resampling procedure (1,000 iterations) is performed to evaluate the robustness of the assigned trend class, expressed as the percentage of iterations that reproduce the original classification.

7. **Permutation significance testing**
   Temporal labels are randomly permuted 1,000 times to estimate the probability that the observed trend magnitude could arise by chance.

8. **Threshold sensitivity analysis**
   Conservative, baseline, and permissive threshold scenarios are evaluated to quantify the agreement of trend classifications with the baseline configuration, reproducing the sensitivity assessment reported in the manuscript.

9. **Demonstration dataset generation**
   When no input file is provided, the script automatically generates a synthetic station-year dataset based on network-wide annual means and spatial variability patterns reported in the study.

10. **Results generation**
    The script produces a CSV file containing:

* Number of observations;
* Raw and normalized Theil–Sen slopes;
* Kendall's tau coefficient;
* p-values;
* Inter-period median differences;
* Trend classifications;
* Bootstrap stability percentages;
* Permutation p-values; and
* Statistical significance indicators.
