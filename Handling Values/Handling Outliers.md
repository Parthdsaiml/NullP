| **Data Type**              | **Transformation Method**                         | **When to Use**                                                                 |
|----------------------------|---------------------------------------------------|---------------------------------------------------------------------------------|
| **Positively Skewed**       | Log Transformation                               | When values are strictly positive and there is a long right tail (positive skew). |
|                            | Box-Cox Transformation                           | If the data is strictly positive and you want to fine-tune the transformation.  |
| **Negatively Skewed**       | Box-Cox Transformation                           | When values are strictly positive and you need to handle left-skewed data.      |
| **Data with Zero or Negative Values** | Yeo-Johnson Transformation                 | When your data contains negative values or zero.                                |
| **Count Data (Moderate Skew)** | Square Root Transformation                     | When the data is count data (positive integers) and moderately skewed.         |
| **Extreme Outliers**        | Reciprocal Transformation                        | When there are extreme positive outliers and a few very large values.           |
| **Normal Distribution (No Skew)** | No Transformation (or Z-score for outliers) | When data is already symmetric or normal, no transformation needed.            |
