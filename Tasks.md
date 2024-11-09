| **Step**                          | **Get** | **Do** | **Visualize** | **GDV** |
|------------------------------------|---------|--------|---------------|---------|
| 1. **Remove duplicates**           | [ ]     | [ ]    | [ ]           | [ ]     |
| 2. **Handle missing values**       | [x]     | [x]    | [x]           | [x]     |
| 3. **Fix data types**             | [ ]     | [ ]    | [ ]           | [ ]     |
| 4. **Handle outliers**             | [x]     | [_]    | [x]           | [ ]     |
| 5. **Remove irrelevant features**  | [ ]     | [ ]    | [ ]           | [ ]     |
| 6. **Normalize or standardize**   | [ ]     | [ ]    | [ ]           | [ ]     |
| 7. **Correct inconsistent data**  | [ ]     | [ ]    | [ ]           | [ ]     |
| 8. **Convert categorical data**   | [ ]     | [ ]    | [ ]           | [ ]     |
| 9. **Create new features**        | [ ]     | [ ]    | [ ]           | [ ]     |
| 10. **Check for data leakage**    | [ ]     | [ ]    | [ ]           | [ ]     |

1. **Remove duplicates**: 
   - **What to do**: Identify and remove any repeated rows in the dataset that may result in biased analysis or modeling.

2. **Handle missing values**: 
   - **What to do**: Address missing or null data by either filling (imputation), removing, or otherwise handling gaps to ensure data integrity.

3. **Fix data types**: 
   - **What to do**: Ensure that each feature has the correct data type (e.g., integers, floats, categorical) to perform operations correctly.

4. **Handle outliers**: 
   - **What to do**: Identify and manage extreme values that may distort statistical analyses or modeling (either by removing, transforming, or capping them).

5. **Remove irrelevant features**: 
   - **What to do**: Drop columns or features that are not useful for the analysis or model, as they may introduce noise.

6. **Normalize or standardize**: 
   - **What to do**: Adjust numerical values so they have a consistent scale or distribution, making it easier for algorithms to learn patterns (e.g., scaling between 0 and 1, or standardizing to a mean of 0 and standard deviation of 1).

7. **Correct inconsistent data**: 
   - **What to do**: Address data inconsistencies or errors, such as mismatched categories (e.g., "yes" vs "y" vs "1") or inconsistent naming conventions.

8. **Convert categorical data**: 
   - **What to do**: Convert non-numeric categorical data into numeric representations, such as using one-hot encoding or label encoding, so it can be used in machine learning models.

9. **Create new features**: 
   - **What to do**: Derive new variables from existing ones that could provide additional predictive power (e.g., creating an "age group" from a raw age column).

10. **Check for data leakage**: 
    - **What to do**: Ensure that the training data does not inadvertently contain information from the future or from the test set that could bias model performance.

