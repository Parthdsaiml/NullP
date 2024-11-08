| **Condition**                       | **Method**                                   | **When to Use**                                                                 |
|-------------------------------------|----------------------------------------------|---------------------------------------------------------------------------------|
| **Target Column (Numeric)**         | **IQR (Interquartile Range) Outliers**        | Use when the data is highly skewed (|skewness| > 0.5), indicating potential extreme values. |
|                                     | **Standard Deviation (SD) Outliers**         | Use when the data is approximately symmetric (|skewness| ≤ 0.5), typically for detecting moderate outliers. |
| **Other Numeric Columns**           | **IQR (Interquartile Range) Outliers**        | Use when column has high skewness (|skewness| > 0.5), indicating significant deviation from normal distribution. |
|                                     | **Standard Deviation (SD) Outliers**         | Use when column is nearly symmetric (|skewness| ≤ 0.5), for detecting small/moderate deviations. |
| **Categorical Columns**             | **Categorical Outlier Detection**            | Use when the column contains categorical data (e.g., string or object type). Detects unusual frequency distributions or rare categories. |
