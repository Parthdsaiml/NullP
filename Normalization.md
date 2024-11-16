--------------------------------------------------------------
                      Min-Max Scaling
--------------------------------------------------------------
Formula:

    X_scaled = (X - X_min) / (X_max - X_min)

Explanation:
- Min-Max Scaling scales the data to a fixed range, typically between 0 and 1.
- X_min: The minimum value in the column.
- X_max: The maximum value in the column.
- Useful when you need the features to be within a specific range (e.g., for neural networks).
--------------------------------------------------------------

--------------------------------------------------------------
                       Z-Score Normalization
--------------------------------------------------------------
Formula:

    X_scaled = (X - μ) / σ

Where:
- μ (mean): The average of the column.
- σ (standard deviation): The spread of the values in the column.

Explanation:
- Z-Score Normalization transforms the data such that it has a mean of 0 and a standard deviation of 1.
- Useful for algorithms that assume the data is normally distributed, like linear regression or KNN.
--------------------------------------------------------------

--------------------------------------------------------------
                       Robust Scaling
--------------------------------------------------------------
Formula:

    X_scaled = (X - Median) / IQR

Where:
- Median: The middle value of the column.
- IQR (Interquartile Range): The range between the 25th percentile (Q1) and the 75th percentile (Q3).

Explanation:
- Robust Scaling uses the median and IQR, making it more robust to outliers.
- Best used when the data contains significant outliers.
--------------------------------------------------------------

--------------------------------------------------------------
                       Log Transformation
--------------------------------------------------------------
Formula:

    X_scaled = log(X)

Explanation:
- Log Transformation is applied to compress the range of values and make the distribution more symmetric.
- Typically used when the data is highly skewed or when values span several orders of magnitude.
- Note: Only works with positive values of X. For non-positive values, a constant must be added to make them positive.
--------------------------------------------------------------

--------------------------------------------------------------
                       Decimal Scaling
--------------------------------------------------------------
Formula:

    X_scaled = X / 10^j

Where:
- j is the smallest integer such that max(|X_scaled|) < 1.

Explanation:
- Decimal Scaling involves dividing the data by a power of 10 to bring the magnitude of values between -1 and 1.
- Used when you want to scale the data by adjusting its magnitude.
--------------------------------------------------------------

--------------------------------------------------------------
                       Unit Vector Scaling
--------------------------------------------------------------
Formula:

    X_scaled = X / ||X||

Where:
- ||X|| is the Euclidean norm (magnitude) of the vector X:

    ||X|| = sqrt( Σ(X_i)^2 )

Explanation:
- Unit Vector Scaling normalizes the data so that each feature has a magnitude of 1.
- Useful for distance-based algorithms like KNN or for comparing vectors in high-dimensional space.
--------------------------------------------------------------
