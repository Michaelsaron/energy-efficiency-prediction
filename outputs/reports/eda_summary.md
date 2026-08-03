# Exploratory Data Analysis Summary

Generated: 2026-08-03 13:09:18

## Correlation with Heating Load

| Feature                   |   Correlation |
|:--------------------------|--------------:|
| Cooling_Load              |    0.975862   |
| Overall_Height            |    0.889431   |
| Roof_Area                 |   -0.861828   |
| Surface_Area              |   -0.65812    |
| Relative_Compactness      |    0.622272   |
| Wall_Area                 |    0.455671   |
| Glazing_Area              |    0.269841   |
| Glazing_Area_Distribution |    0.0873676  |
| Orientation               |   -0.00258653 |

## Feature Skewness

| Feature                   |   Skewness |
|:--------------------------|-----------:|
| Wall_Area                 |  0.533417  |
| Relative_Compactness      |  0.495513  |
| Cooling_Load              |  0.395992  |
| Heating_Load              |  0.360449  |
| Roof_Area                 | -0.162764  |
| Surface_Area              | -0.125131  |
| Glazing_Area_Distribution | -0.0886892 |
| Glazing_Area              | -0.0602542 |
| Overall_Height            |  0         |
| Orientation               |  0         |

## IQR Outlier Report

The IQR analysis identifies unusual values for review. Values are not
automatically removed because they may represent valid building configurations.

| Feature                   |   Outlier Count |   Outlier Percentage |   Lower IQR Bound |   Upper IQR Bound |
|:--------------------------|----------------:|---------------------:|------------------:|------------------:|
| Relative_Compactness      |               0 |                    0 |            0.4613 |            1.0513 |
| Surface_Area              |               0 |                    0 |          404.25   |          943.25   |
| Wall_Area                 |               0 |                    0 |          220.5    |          416.5    |
| Roof_Area                 |               0 |                    0 |           21.4375 |          339.938  |
| Overall_Height            |               0 |                    0 |           -1.75   |           12.25   |
| Orientation               |               0 |                    0 |            0.5    |            6.5    |
| Glazing_Area              |               0 |                    0 |           -0.35   |            0.85   |
| Glazing_Area_Distribution |               0 |                    0 |           -1.625  |            7.375  |
| Heating_Load              |               0 |                    0 |          -15.02   |           59.68   |
| Cooling_Load              |               0 |                    0 |          -10.6487 |           59.4012 |

## Generated EDA Figures

- `outputs/figures/best_model_residual_distribution.png`

## Main Interpretation

- Features with larger absolute correlations have stronger linear relationships
  with heating load.
- Strong skewness may indicate non-normal distributions or discrete design
  categories.
- Reported outliers should be evaluated in context before capping or removal.
- Tree and boosting models can capture nonlinear relationships that correlation
  alone may not reveal.
