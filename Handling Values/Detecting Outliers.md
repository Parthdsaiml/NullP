
| **Condition**                       | **Method**                                   | **When to Use**                                                                 |
|-------------------------------------|----------------------------------------------|---------------------------------------------------------------------------------|
| **Target Column (Numeric)**         | **IQR (Interquartile Range) Outliers**        | Use when the data is highly skewed (|skewness| > 0.5), indicating potential extreme values. |
|                                     | **Standard Deviation (SD) Outliers**         | Use when the data is approximately symmetric (|skewness| ≤ 0.5), typically for detecting moderate outliers. |
| **Other Numeric Columns**           | **IQR (Interquartile Range) Outliers**        | Use when column has high skewness (|skewness| > 0.5), indicating significant deviation from normal distribution. |
|                                     | **Standard Deviation (SD) Outliers**         | Use when column is nearly symmetric (|skewness| ≤ 0.5), for detecting small/moderate deviations. |
| **Categorical Columns**             | **Categorical Outlier Detection**            | Use when the column contains categorical data (e.g., string or object type). Detects unusual frequency distributions or rare categories. |

### Key Insights:
1. **Skewed Numeric Data**:
   - If the data is highly skewed (positively or negatively), the function opts for IQR-based detection to identify extreme values that deviate significantly from the central mass of data.
   - If the data is normally distributed or symmetric (low skew), standard deviation-based detection is used to identify values that are far from the mean (typically 1.5 or more standard deviations).

2. **Categorical Data**:
   - For categorical columns, the function looks for outliers based on frequency or the occurrence of rare/unexpected categories.
   - This method doesn't rely on skewness, as it's not relevant for categorical data, but rather the distribution of categories across the dataset.

3. **Target Column**:
   - Special handling is done for the target column (usually the dependent variable in a machine learning model), where different strategies (IQR or SD) are selected based on skewness to best handle outliers before modeling.

### Summary of Methods:
- **IQR (Interquartile Range)**: Good for data with significant skew or extreme values, as it identifies values that lie far outside the normal range.
- **Standard Deviation**: Best for symmetric or normal data, where outliers are defined as values that lie a few standard deviations away from the mean.
- **Categorical Outlier Detection**: Useful for detecting rare categories or unexpected frequency distributions in categorical data.

This table highlights when each method of outlier detection is used based on the skewness of the data type, aligning with the logic in your provided function.
