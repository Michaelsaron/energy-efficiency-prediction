# Data Understanding Report

Generated: 2026-08-03 13:09:18

## Dataset Overview

- Rows: **768**
- Columns: **10**
- Target variable: **Heating_Load**
- Duplicate rows: **0**
- Total missing values: **0**

## Column Summary

| Column                    | Data Type   |   Missing Values |   Unique Values |
|:--------------------------|:------------|-----------------:|----------------:|
| Relative_Compactness      | float64     |                0 |              12 |
| Surface_Area              | float64     |                0 |              12 |
| Wall_Area                 | float64     |                0 |               7 |
| Roof_Area                 | float64     |                0 |               4 |
| Overall_Height            | float64     |                0 |               2 |
| Orientation               | int64       |                0 |               4 |
| Glazing_Area              | float64     |                0 |               4 |
| Glazing_Area_Distribution | int64       |                0 |               6 |
| Heating_Load              | float64     |                0 |             586 |
| Cooling_Load              | float64     |                0 |             636 |

## Descriptive Statistics

| Feature                   |   count |       mean |       std |    min |      25% |    50% |      75% |    max |
|:--------------------------|--------:|-----------:|----------:|-------:|---------:|-------:|---------:|-------:|
| Relative_Compactness      |     768 |   0.764167 |  0.105777 |   0.62 |   0.6825 |   0.75 |   0.83   |   0.98 |
| Surface_Area              |     768 | 671.708    | 88.0861   | 514.5  | 606.375  | 673.75 | 741.125  | 808.5  |
| Wall_Area                 |     768 | 318.5      | 43.6265   | 245    | 294      | 318.5  | 343      | 416.5  |
| Roof_Area                 |     768 | 176.604    | 45.166    | 110.25 | 140.875  | 183.75 | 220.5    | 220.5  |
| Overall_Height            |     768 |   5.25     |  1.75114  |   3.5  |   3.5    |   5.25 |   7      |   7    |
| Orientation               |     768 |   3.5      |  1.11876  |   2    |   2.75   |   3.5  |   4.25   |   5    |
| Glazing_Area              |     768 |   0.234375 |  0.133221 |   0    |   0.1    |   0.25 |   0.4    |   0.4  |
| Glazing_Area_Distribution |     768 |   2.8125   |  1.55096  |   0    |   1.75   |   3    |   4      |   5    |
| Heating_Load              |     768 |  22.3072   | 10.0902   |   6.01 |  12.9925 |  18.95 |  31.6675 |  43.1  |
| Cooling_Load              |     768 |  24.5878   |  9.51331  |  10.9  |  15.62   |  22.08 |  33.1325 |  48.03 |
