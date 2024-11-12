### Properties of Irrelevant or Non-Relevant Columns

1. **Constant Columns**  
   - Columns with the same value across all rows.

2. **High Cardinality with Low Frequency**  
   - Categorical columns with too many unique, infrequent values.

3. **Highly Skewed Numerical Features**  
   - Numerical columns with a non-normal distribution (e.g., heavy skew).

4. **Duplicate Columns**  
   - Columns that are exact duplicates of others.

5. **ID Columns**  
   - Columns like `user_id`, `transaction_id` that are unique identifiers.

6. **Columns with Many Missing Values**  
   - Columns with a high percentage of missing or null values.

7. **Irrelevant Text Columns**  
   - Free-text fields like `comments` or `remarks` without structured data.

8. **Columns with Too Many Unique Values (Sparse Data)**  
   - Columns with too many unique values, like `order_id`, where each value is unique.

9. **Feature Leakage**  
   - Columns that directly or indirectly encode the target variable.

10. **Redundant Time Features**  
    - Multiple columns encoding the same time-related information.

11. **Multicollinearity**  
    - Highly correlated columns that provide redundant information.

12. **Columns with Outliers or Errors**  
    - Columns with extreme or erroneous values that do not represent the data properly.
