import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
df.info()
unique_identifiers = [
        "RowNumber",
        "ID",
        "UserID",
        "UserId",
        "CustomerId",
    
        "CustomerID",
        "ProductID",
        "EmployeeID",
        "StudentID",
        "AccountNumber",
        "TransactionID",
        "OrderID",
        "InvoiceNumber",
        "SessionID",
        "ActivityID",
        "VisitID",
        "UUID",
        "UUID(UniversallyUniqueIdentifier)",
        "SocialSecurityNumber(SSN)",
        "DriversLicenseNumber",
        "BusinessLicenseNumber",
        "LoyaltyProgramID",
        "MembershipID",
        "RegistrationNumber",
        "RollNumber",
        "User ID",
        "Customer ID",
        "Product ID",
        "Employee ID",
        "Student ID",
        "Account Number",
        "Transaction ID",
        "Order ID",
        "Invoice Number",
        "Session ID",
        "Activity ID",
        "Visit ID",
        "UUID (Universally Unique Identifier)",
        "Social Security Number (SSN)",
        "Driver's License Number",
        "Business License Number",
        "Loyalty Program ID",
        "Membership ID",
        "Registration Number",
        "Roll Number",
    ]
usefullColumns = [
    'Revenue',
    'Customer Satisfaction Score',
    'Net Promoter Score',
    'Sales Volume',
    'Churn Rate',
    'Patient Age',
    'Blood Pressure',
    'Cholesterol Level',
    'Body Mass Index',
    'Blood Sugar Level',
    'Temperature',
    'Humidity',
    'Air Quality Index',
    'Precipitation Levels',
    'Wind Speed',
    'Income Level',
    'Education Level',
    'Employment Status',
    'Age',
    'Marital Status',
    'Product Ratings',
    'User Engagement Metrics',
    'Inventory Levels',
    'Market Share',
    'Cost per Acquisition',
    'Geographic Location',
    'Date',
    'Product Category',
    'Customer Segment'
]




class NullCheck:
    def __init__(self, df):
        self.df = df.copy()  # Store a copy of the DataFrame
        self.droppingColumns = []  # Initialize the list of columns to drop

        
    
    # Null Check: Check for missing values and display summary
    def nullCheck(self):
        totalNullsCombineColumns = self.df.isnull().sum().sum()
        if totalNullsCombineColumns != 0:
            print("Total Nulls: ", totalNullsCombineColumns)
            print("Null Count in Each Column:")
            print(self.df.isnull().sum())
        print("No Null Values found")
        
    def isCategorical(self, column):
        return self.df[column].dtype == 'object'

    # Automate removing missing values with the threshold for missing data
    def automateRemovingNullValues(self, threshold=0.1):
        for column in self.df.columns:
            # Skip columns that are too large
            if len(self.df[column]) > 1000000:
                continue  # Need a better approach to handle very large columns

            # Handle categorical columns (replace NaNs with Mode)
            if self.isCategorical(column):
                self.replaceWithMode(column)
            else:
                # Check if the column is numeric (not categorical)
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    if self.skewCheck(self.df[column]):
                        # If the column is skewed, replace NaNs with Median
                        self.replaceWithMedian(column)
                    else:
                        # For non-skewed numeric columns, replace NaNs with Mean
                        self.replaceWithMean(column)
                else:
                    # Handle non-numeric columns
                    if self.df[column].isnull().all():
                        self.df[column].fillna('Unknown', inplace=True)
                    else:
                        self.replaceWithMode(column)
        return self.df

    # Check for skewness or kurtosis in a numeric column
    def skewCheck(self, column, skew_threshold=0.5, kurtosis_threshold=3.0):
        if not pd.api.types.is_numeric_dtype(column):
            return False
        skewness = column.skew()
        kurt = column.kurtosis()
        skewed = abs(skewness) > skew_threshold
        heavy_tailed = abs(kurt) > kurtosis_threshold
        return skewed or heavy_tailed

    # Replace missing values with Mode (for categorical data)
    def replaceWithMode(self, column):
        modeValue = self.df[column].mode()[0]  # Get the most frequent value
        self.df[column] = self.df[column].fillna(modeValue)
        return self.df

    # Replace missing values with Mean (for numeric data)
    def replaceWithMean(self, column):
        meanValue = self.df[column].mean()
        self.df[column] = self.df[column].fillna(meanValue)
        return self.df

    # Replace missing values with Median (for skewed numeric data)
    def replaceWithMedian(self, column):
        medianValue = self.df[column].median()
        self.df[column] = self.df[column].fillna(medianValue)
        return self.df



    # Automate the process of handling missing values
        
    

    # Check Linearity: Using OLS to check for linearity with the target column
    # def check_linearity(self, target_column, graphs=False):
    #     y = self.df[target_column]
    #     X = self.df.drop(columns=[target_column])
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X).fit()
    #     r_squared = model.rsquared
    #     p_values = model.pvalues
    #     if r_squared < 0.5:  # Threshold for weak linearity
    #         return False
    #     if any(p_values > 0.05):  # Check for statistical significance of p-values
    #         return False
    #     if graphs:
    #         residuals = model.resid
    #         plt.figure(figsize=(8, 6))
    #         sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red'})
    #         plt.title('Residual Plot')
    #         plt.xlabel('Fitted Values')
    #         plt.ylabel('Residuals')
    #         plt.show()
    #     return True

    # Replacing simpler data null values using Mode, Median, Mean or KNN
    def replacingSimplerDataNullValues(self, percentNull=0.4, dropColumns=True):
        for column in self.df.columns:
            nullCount = self.df[column].isnull().sum()
            nullPercent = nullCount / len(self.df[column])
            
            if nullPercent > percentNull:
                self.droppingColumns.append(column)
                continue

            if pd.api.types.is_string_dtype(self.df[column]):
                self.df = self.replaceWithMode(column)
            elif pd.api.types.is_numeric_dtype(self.df[column]):
                if self.skewCheck(self.df[column]):
                    self.df = self.replaceWithMedian(column)
                else:
                    self.df = self.replaceWithMean(column)

        # Drop columns if specified
        if dropColumns:
            if self.droppingColumns:
                self.df = self.df.drop(columns=self.droppingColumns)
                print(f"Dropped columns with high null values: {self.droppingColumns}")
            else:
                print("No columns were dropped.")
        
        return self.df
    def visualizeMissingData(self, heatmap_color='viridis', save_fig=False, fig_prefix='missing_data'):
        if self.df.empty:
            print("DataFrame is empty.")
            return
        
        # Create a boolean DataFrame for missing values (True = missing)
        missing_data = self.df.isnull()
        
        # Heatmap of missing values
        plt.figure(figsize=(12, 8))
        sns.heatmap(missing_data, cmap=heatmap_color, cbar=False, yticklabels=False, xticklabels=self.df.columns)
        plt.title('Missing Data Heatmap', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Observations', fontsize=14)
        
        # Save the figure if requested
        if save_fig:
            plt.savefig(f"{fig_prefix}_heatmap.png", bbox_inches='tight')
        
        plt.show()

        # Bar chart showing the count of missing values per feature
        missing_counts = self.df.isnull().sum()
        
        plt.figure(figsize=(10, 5))
        missing_counts.plot(kind='bar', color='skyblue')
        plt.title('Missing Values per Feature', fontsize=16)
        plt.ylabel('Number of Missing Values', fontsize=14)
        plt.xlabel('Features', fontsize=14)
        plt.xticks(rotation=45)
        
        # Adding percentages to the bar chart
        for index, value in enumerate(missing_counts):
            plt.text(index, value, f'{value} ({value/len(self.df)*100:.1f}%)', ha='center', va='bottom')

        # Save the figure if requested
        if save_fig:
            plt.savefig(f"{fig_prefix}_missing_counts.png", bbox_inches='tight')
        
        plt.show()



def isCategorical(df, column_name, cardinality_threshold=0.1):
    dtype = df[column_name].dtype
    
    # Check if the column is already a pandas Categorical type
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    
    # Check if the column is of object dtype (i.e., strings or mixed types)
    if pd.api.types.is_object_dtype(dtype):
        unique_count = len(df[column_name].unique())
        total_count = len(df)
        
        # Heuristic: if unique values are less than the cardinality threshold, consider it categorical
        if unique_count < cardinality_threshold * total_count:
            return True
        
    # Check for Boolean columns (often treated as categorical)
    if pd.api.types.is_bool_dtype(dtype):
        return True
    
    # If none of the above, return False
    return False









class OutlierDetection:
    def __init__(self, df, target_column):
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying original data
        self.target_column = target_column
        
    def isolation_forest_outliers(self, column, contamination=0.1):
        """
        Detect outliers using the Isolation Forest method.
        """
        data = column.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data)
        return np.where(outliers == -1)[0].tolist()

    
    def getIQRRange(self, column, dynamicValue = -1):
        """
        Calculate the IQR (Interquartile Range) and dynamic range for outlier detection.
        """
        sortedData = np.sort(column)
        if len(sortedData) <= 1:
            return [sortedData[0], sortedData[0]]  # If only 1 or 0 elements, no IQR calculation

        Q1 = np.percentile(sortedData, 25)
        Q3 = np.percentile(sortedData, 75)
        IQR = Q3 - Q1
        lowerBound = Q1 - (dynamicValue if dynamicValue != -1 else 1.5) * IQR
        upperBound = Q3 + (dynamicValue if dynamicValue != -1 else 1.5) * IQR
        return [lowerBound, upperBound]

    def iqrOutliers(self, column, valueDynamic=-1):
        """
        Identify outliers in a column based on IQR.
        """
        iqrRange = self.getIQRRange(column, valueDynamic)
        outlier_indices = [idx for idx, value in enumerate(column) if value < iqrRange[0] or value > iqrRange[1]]
        return outlier_indices

    def sdRange(self, column, dynamicValue=-1):
        """
        Calculate the SD (Standard Deviation) range for outlier detection.
        """
        meanValue = column.mean()
        stdValue = column.std()
        lowerRange = meanValue - (dynamicValue if dynamicValue != -1 else 3) * stdValue
        upperRange = meanValue + (dynamicValue if dynamicValue != -1 else 3) * stdValue
        return [lowerRange, upperRange]

    def sdOutliers(self, column, valueDynamic=-1):
        """
        Identify outliers in a column based on Standard Deviation.
        """
        rangeSd = self.sdRange(column, valueDynamic)
        outlierIndices = [idx for idx, value in enumerate(column) if value < rangeSd[0] or value > rangeSd[1]]
        return outlierIndices




    

    def skewedDetection(self):
        """
        Detect skewed columns in the dataframe.
        Returns a list of skewness values for numeric columns.
        """
        skewedList = []
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if self.df[column].nunique() < 5:  # Ignore very low cardinality columns
                    skewedList.append(None)
                    continue
                skew_value = self.df[column].skew()
                if skew_value > 0.5:
                    skewedList.append(1)  # Right skewed
                elif skew_value < -0.5:
                    skewedList.append(-1)  # Left skewed
                else:
                    skewedList.append(0)  # Not skewed
            else:
                skewedList.append(None)  # For non-numeric columns
        return skewedList

    # Returns True if there are outliers, False otherwise

    # Existing methods from the previous part would be here...



    def showOutliers(self, plot_type="boxplot"):
    # Ensure plot_type is a string and is valid
        if not isinstance(plot_type, str):
            raise ValueError("plot_type must be a string")

        if plot_type not in ["boxplot", "scatter", "histogram"]:
            raise ValueError(f"Invalid plot_type: {plot_type}. Supported plot types are ['boxplot'].")
    
        for column in self.df.columns:
        # Only plot numeric columns
            if self.df[column].dtype in ['float64', 'int64']:
                plt.figure(figsize=(5, 3))

                if plot_type == "boxplot":
                    sns.boxplot(y=self.df[column])
                    plt.title(f"Box Plot of {column}")
                    plt.ylabel(column)  # Fixed the typo here
                elif plot_type == 'scatter':
                    plt.scatter(self.df.index, self.df[column])
                    plt.title(f'Scatter Plot of {column}')
                    plt.ylabel(column)
                    plt.xlabel('Index')
                elif plot_type == 'histogram':
                    sns.histplot(self.df[column], bins=30, kde=True)
                    plt.title(f'Histogram of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                
                plt.show()


            
    def showColumnOutliers(self, column, plot_type='boxplot'):
        # Check if the column exists in the DataFrame
        if column not in self.df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return
        
        plt.figure(figsize=(8, 6))
        
        if plot_type == 'boxplot':
            # Create a box plot
            sns.boxplot(y=self.df[column])
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)

        elif plot_type == 'scatter':
            # Create a scatter plot
            plt.scatter(self.df.index, self.df[column])
            plt.title(f'Scatter Plot of {column}')
            plt.ylabel(column)
            plt.xlabel('Index')

        elif plot_type == 'histogram':
            # Create a histogram
            sns.histplot(self.df[column], bins=30, kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

        else:
            print(f"Plot type '{plot_type}' is not supported.")
            return
        
        plt.show()
    def detectOutliersIndex(self, count=False):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()

                # Skip columns with constant values (no variance)
                if data.nunique() == 1:
                    continue

                # Calculate IQR for the column
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1

                # Define the bounds for outliers (1.5 * IQR rule)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify the outliers in the column based on index
                outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index

                # If outliers exist, store either the count or the list of outliers' indices
                if not outlier_indices.empty:
                    if count:
                        all_outliers[column] = len(outlier_indices)  # Store count of outliers
                    else:
                        all_outliers[column] = outlier_indices.tolist()  # Store list of outlier indices
                else:
                    # If no outliers detected, add an empty list or a placeholder
                    all_outliers[column] = []

            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, detect rare categories as outliers
                value_counts = self.df[column].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < 0.01].index.tolist()  # Less than 1% frequency

                # Identify indices where rare categories appear
                rare_indices = self.df[column][self.df[column].isin(rare_categories)].index

                # If rare categories exist, add their indices as outliers
                if not rare_indices.empty:
                    if count:
                        all_outliers[column] = len(rare_indices)  # Store count of rare category indices
                    else:
                        all_outliers[column] = rare_indices.tolist()  # Store list of indices for rare categories
                else:
                    # If no rare categories, add an empty list or a placeholder
                    all_outliers[column] = []

        return all_outliers

    
    def detectOutliers(self, count=False):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            # Skip non-numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()

                # Skip columns with constant values (no variance)
                if data.nunique() <= 2:
                    continue
                print(column)
                # Calculate IQR for the column
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1

                # Define the bounds for outliers (1.5 * IQR rule)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify the outliers in the column
                outliers = data[(data < lower_bound) | (data > upper_bound)]

                # If outliers exist, store either the count or the list of outliers
                if not outliers.empty:
                    if count:
                        all_outliers[column] = len(outliers)  # Store count of outliers
                    else:
                        all_outliers[column] = outliers.tolist()  # Store list of outliers
                else:
                    # If no outliers detected, add an empty list or a placeholder
                    all_outliers[column] = []

            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, detect rare categories as outliers
                value_counts = self.df[column].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < 0.01].index.tolist()  # Less than 1% frequency

                # If rare categories exist, add them as outliers
                if rare_categories:
                    if count:
                        all_outliers[column] = len(rare_categories)  # Store count of rare categories
                    else:
                        all_outliers[column] = rare_categories  # Store the list of rare categories
                else:
                    # If no rare categories, add an empty list or a placeholder
                    all_outliers[column] = []

        return all_outliers

    def removeOutliers(self):
        
        all_outliers = self.detectOutliersIndex(count=False)

        # Collect all the outlier indices across all columns
        outlier_indices_set = set()
        for outliers in all_outliers.values():
            outlier_indices_set.update(outliers)

        # Remove the rows with the outlier indices
        self.df = self.df.drop(index=outlier_indices_set)

        return self.df


        
    def detectCategoricalOutliers(self, column_name, threshold_percent=1):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        column = self.df[column_name]
        category_counts = column.value_counts()
        threshold = len(column) * (threshold_percent / 100)
        outliers = category_counts[category_counts < threshold]
        return outliers.index.tolist(), outliers.values.tolist()  # Return categories and their counts
    
    def detectDynamicOutliers(self):
        allOutliers = {}

        if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
            targetSkewness = self.df[self.target_column].skew()
            if abs(targetSkewness) > 0.5:
                targetOutliers = self.iqrOutliers(self.df[self.target_column])
            else:
                targetOutliers = self.sdOutliers(self.df[self.target_column])

            if (len(targetOutliers) != 0):
                allOutliers[self.target_column] = targetOutliers

        else:
            targetOuliers = self.detectCategoricalOutliers(target_column)
            if (len(targetOutliers) != 0):
                allOutliers[self.target_column] = targetOutliers

        for column in self.df.columns:
            if column == self.target_column:
                continue
            if pd.api.types.is_numeric_dtype(self.df[column]):
                columnSkewness = self.df[column].skew()
                if abs(columnSkewness) > 0.5:
                    outliers = self.iqrOutliers(self.df[column])
                else:
                    outliers = self.sdOutliers(self.df[column])
            elif (pd.api.types.is_object_dtype(self.df[column])):
                outliers = self.detectCategoricalOutliers(column)
                if (len(outliers) != 0):
                    allOutliers[column] = outliers
        return allOutliers
    def handleOutliers(self, series, outliers, method="impute", lower_bound=None, upper_bound=None):
        if len(outliers) > 0:
            if method == "remove":
            # Option 1: Remove outliers
                series = series[~series.isin(outliers)]

            elif method == "cap":
            # Option 2: Cap outliers to a lower or upper bound (e.g., IQR or SD bounds)
                series = series.clip(lower=lower_bound, upper=upper_bound)

            elif method == "impute":
            # Option 3: Impute outliers with a statistic (e.g., mean, median)
                median_value = series.median()
            # Use .loc to safely modify the original DataFrame or Series
                series.loc[series.isin(outliers)] = median_value  # This avoids the SettingWithCopyWarning

            else:
                print("Invalid method specified. Please use 'remove', 'cap', or 'impute'.")
        return series

        
    def automateOutliers(self, way = "impute"):
        allOutliers = self.detectOutliers()
        if self.target_column in allOutliers:
            if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                targetOutliers = allOutliers[self.target_column]
                self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers, way)

        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if column in allOutliers:
                    columnOutliers = allOutliers[column]
                    self.df[column] = self.handleOutliers(self.df[column], columnOutliers, way)
            
        return self.df
      
    # def automateOutliersAndNormalisation(self, target = False, columnC = False):
    #     allOutliers = self.detectOutliers()
        
    #     # Step 1: Handle outliers in the target column
    #     if self.target_column in allOutliers:
    #         targetOutliers = allOutliers[self.target_column]
    #         self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers)

    #     # Step 2: Handle outliers in other columns
    #     for column in self.df.columns:
    #         if column != self.target_column and column in allOutliers:
    #             columnOutliers = allOutliers[column]
    #             self.df[column] = self.handleOutliers(self.df[column], columnOutliers)

    #     # Step 3: Apply transformations to normalize the data
    #     if (target and columnC):
    #         self.apply_transformation()
    #     elif (target):
    #         self.apply_transformationJustTarget()
    #     elif (columnC):
    #         self.apply_transformation()
    #     else:
    #         self.apply_transformation()
    
     
    # def apply_transformationJustTarget(self):
    #     data = self.df[self.target_column]  # Get the target column from the DataFrame
    #     skewness = stats.skew(data)
        
    #     # Step 1: Check for negative or zero values and apply Yeo-Johnson if needed
    #     if np.any(data <= 0):
    #         pt = PowerTransformer(method='yeo-johnson')
    #         self.df[self.target_column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()  # Apply Yeo-Johnson

    #     # After Yeo-Johnson, re-check skewness
    #     data = self.df[self.target_column]  # Re-get the target column after transformation
    #     skewness = stats.skew(data)
        
    #     # Step 2: Apply transformations based on skewness
    #     if skewness > 1:  # Positively skewed data
    #         self.df[self.target_column] = np.log(data + 1)  # Log transformation (adding 1 to handle zero values)
    #     elif skewness < -1:  # Negatively skewed data
    #         # Box-Cox requires strictly positive values
    #         self.df[self.target_column], _ = stats.boxcox(data[data > 0])  # Filter out non-positive values for Box-Cox
    #     elif 0 < skewness <= 1:  # Moderately positively skewed data
    #         self.df[self.target_column] = np.sqrt(data)  # Square root transformation
    #     elif -1 <= skewness < 0:  # Moderately negatively skewed data
    #         # Box-Cox transformation for moderately skewed negative data (requires positive values)
    #         self.df[self.target_column], _ = stats.boxcox(data[data > 0])
    #     # After transformation, re-check normality
    #     return self.check_normality()
        
    # def applyTransofmation(self):
    #     for column in self.df.columns:
    #         data = self.df[column]
    #         skewness = stats.skew(data)
    #         # Apply Yeo-Johnson if there are negative or zero values
    #         if np.any(data <= 0):
    #             pt = PowerTransformer(method='yeo-johnson')
    #             self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    #         # After transformation, re-check skewness and apply appropriate transformations
    #         data = self.df[column]
    #         skewness = stats.skew(data)
    #         # Apply transformations based on skewness
    #         if skewness > 1:  # Positively skewed data
    #             self.df[column] = np.log(data + 1)
    #         elif skewness < -1:  # Negatively skewed data
    #             self.df[column], _ = stats.boxcox(data[data > 0])
    #         elif 0 < skewness <= 1:  # Moderately positively skewed
    #             self.df[column] = np.sqrt(data)
    #         elif -1 <= skewness < 0:  # Moderately negatively skewed
    #             self.df[column], _ = stats.boxcox(data[data > 0])
    #     return self.check_normality()
        
    # def applyTransformationExceptTarget(self):
    # # Apply transformations to all columns, except the target column
    #     for column in self.df.columns:
    #         if column != self.target_column:  # Skip the target column
    #             data = self.df[column]
    #             skewness = stats.skew(data)

    #         # Apply Yeo-Johnson if there are negative or zero values
    #             if np.any(data <= 0):
    #                 pt = PowerTransformer(method='yeo-johnson')
    #                 self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    #         # After transformation, re-check skewness and apply appropriate transformations
    #             data = self.df[column]
    #             skewness = stats.skew(data)
    #         # Apply transformations based on skewness
    #             if skewness > 1:  # Positively skewed data
    #                 self.df[column] = np.log(data + 1)
    #             elif skewness < -1:  # Negatively skewed data
    #                 self.df[column], _ = stats.boxcox(data[data > 0])
    #             elif 0 < skewness <= 1:  # Moderately positively skewed
    #                 self.df[column] = np.sqrt(data)
    #             elif -1 <= skewness < 0:  # Moderately negatively skewed
    #                 self.df[column], _ = stats.boxcox(data[data > 0])
    #     return self.check_normality()
        
    def check_normality(self):
        for column in self.df.columns:
            data = self.df[column]
            normality_test = NormalityTest(data)
            is_normal = normality_test.check_normality()
            if is_normal:
                continue
            else:
                return False
        return True

     
    
        



class FixDataTypes:
    def __init__(self):
        """Initialize with a DataFrame."""
        pass
    def replace_strings_with_nan(self,dataFrame, column):
        
        if column in dataFrame.columns:
            # Apply a lambda function to replace only string values with NaN
            dataFrame[column] = dataFrame[column].apply(
                lambda x: np.nan if isinstance(x, str) and not x.replace('.', '', 1).isdigit() else x
            )
        else:
            print(f"Column '{column}' does not exist in the dataframe.")
        return dataFrame
    def getCompoundLinearity(self, dataFrame=None):
        """
        This method processes each column of the DataFrame and returns a list of tuples with column names 
        and their respective linearity details (percentages of numeric, object, and null values).
        """
        # If no DataFrame is passed, use the one from the class instance
        if dataFrame is None:
            dataFrame = self.dataFrame
        
        compoundList = []  # List to hold the linearity data for each column
        for column in dataFrame.columns:
            linearity = self.getLinearity(dataFrame, column, False)
            compoundList.append((column, linearity))  # Append the column name and its linearity data
        
        # Check and print the compound list to confirm it's correctly populated
        return compoundList  # Return the list of tuples containing column names and their linearity data
    
    def getLinearity(self, dataFrame, column, get=True):
        
        """
        This method computes the percentage of numeric, object, and null data for a given column.
        """
        isString = dataFrame[column].dtype == "object"
        linearList = []
        tColumn = pd.to_numeric(dataFrame[column], errors="coerce")
        
            # Count how many numeric values are in the column (non-NaN values)
        numCount = tColumn.notna().sum()
        objectCount = dataFrame[column].size - numCount
        
            # Calculate the percentages
        numPercentage = numCount / dataFrame[column].size
        objectPercentage = objectCount / dataFrame[column].size
        nullCount = dataFrame[column].isnull().sum() / dataFrame[column].size
        linearList = [numPercentage, objectPercentage, nullCount]
        dataLinearList = ["num", numPercentage * 100, "object", objectPercentage * 100, "null", nullCount * 100]
        

        
        return dataLinearList

            

    
    def showDuplicates(self, dataFrame):
        linearList = self.getCompoundLinearity(dataFrame)
        self.plotLinearityFromList(linearList)
        
    def plotLinearityFromList(self, linearityData):
        """
        This method takes a list of tuples containing column names and their linearity data,
        then generates a stacked bar chart based on the data structure.
        """
        # Check if the linearity data is empty or None
        if linearityData is None or len(linearityData) == 0:
            print("Error: The linearity data is empty or None!")  # Debug print
            return
        
        # Convert the linearity data into a DataFrame for easy plotting
        columns = []
        num_percentage = []
        object_percentage = []
        null_percentage = []

        for column, data in linearityData:
            columns.append(column)
            # Process the linearity data
            for i in range(0, len(data), 2):
                type_value = data[i]
                value = data[i + 1]

                if type_value == 'num':
                    num_percentage.append(value)
                elif type_value == 'object':
                    object_percentage.append(value)
                elif type_value == 'null':
                    null_percentage.append(value)

        # Create a DataFrame for plotting the stacked bar chart
        linearity_df = pd.DataFrame({
            'Column': columns,
            'num': num_percentage,
            'object': object_percentage,
            'null': null_percentage
        })

        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        linearity_df.set_index('Column').plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

        ax.set_title('Data Linearity by Column')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Columns')
        ax.legend(title='Data Type')

        plt.tight_layout()
        plt.show()

    def plotMissingValues(self, df, title="Missing Data Heatmap"):
        # Plot heatmap for missing data visualization
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(title)
        plt.show()
       
    def replace_numbers_with_unknown(self, dataFrame, column):
        if column in dataFrame.columns:
        # Use .apply() to transform each value in the column to 'unknown' if it's a number
            dataFrame[column] = dataFrame[column].apply(
                lambda x: 'unknown' if self.is_number(x) else x
            )
        else:
            print(f"Column '{column}' does not exist in the dataframe.")
        return dataFrame

    def is_number(self, x):
        """Helper function to check if a value is a number (int, float, or numeric string)."""
    # Handle NaN or None values
        if pd.isna(x):
            return False
    
    # Check if the value is a number or can be converted to a number
        if isinstance(x, (int, float)): 
            return True
        elif isinstance(x, str):
        # Handle cases like '3.14', '1e3', '-5.2', but exclude non-numeric strings
            try:
                float(x)  # Try to convert the string to a float
                return True
            except ValueError:
                return False  # If it can't be converted to float, it's not a number
    
        return False


    def getSNPercentage(self, dataFrame, column):
        isString = dataFrame[column].dtype == "object"
    
        if isString:
        # Convert to numeric (invalid parsing will be converted to NaN)
            tColumn = pd.to_numeric(dataFrame[column], errors="coerce")
        
        # Count how many numeric values are in the column (non-NaN values)
            numCount = tColumn.notna().sum()
            objectCount = dataFrame[column].size - numCount
        
        # Calculate the percentages
            numPercentage = numCount / dataFrame[column].size
            objectPercentage = objectCount / dataFrame[column].size
        
            print(f"Numeric Percentage: {numPercentage}")
            print(f"String Percentage: {objectPercentage}")

    
    def deepConverting(self, dataFrame, column):
        listOfColumns = []
    # Check if the column dtype is object (string)
        isString = dataFrame[column].dtype == "object"
    
        if isString:
        # Convert to numeric (invalid parsing will be converted to NaN)
            tColumn = pd.to_numeric(dataFrame[column], errors="coerce")
        
        # Count how many numeric values are in the column (non-NaN values)
            numCount = tColumn.notna().sum()
            objectCount = dataFrame[column].size - numCount
        
        # Calculate the percentages
            numPercentage = numCount / dataFrame[column].size
            objectPercentage = objectCount / dataFrame[column].size
        
        # If numeric values are less than 10%, convert all numeric values to "unknown"
            if objectPercentage < 0.1 and objectPercentage > 0.0:
            # Convert numeric values (in string format) to "unknown"
                dataFrame = self.replace_strings_with_nan(dataFrame, column)
        # If string values are less than 10%, convert all string values to NaN
            elif numPercentage < 0.1 and numPercentage > 0.0:
                print(column)
                dataFrame = self.replace_numbers_with_unknown(dataFrame, column)
            # Clean up column values if there are mixed values (str and numeric)
            else:
                listOfColumns.append(column)
                
        if (len(listOfColumns) != 0):
            print(f"{listOfColumns}\nContains unorderd composition of numbers and objects")
        
        return dataFrame

    def convert_to_number(self, value):
        if isinstance(value, str):
        # Try to convert string to float or int
            try:
                if '.' in value:
                    return float(value)  # Convert to float
                return int(value)  # Convert to integer
            except ValueError:
                return value  # Return original string if it cannot be converted
        return value  # If it's already a number (int/float), leave it unchanged


    def convert_column(self, dataFrame, column):
        """Convert a specific column to numeric values where applicable."""
        # Check if the column contains any non-numeric strings
        has_non_numeric = dataFrame[column].apply(lambda x: isinstance(x, str) and not (x.isdigit() or self.is_float(x))).any()
        if (dataFrame[column].dtype == "int64" or dataFrame[column].dtype == "float64"):
            return dataFrame
            
        if has_non_numeric:
            # If any non-numeric string is found, leave the column as is (don't convert to numeric)
            dataFrame = self.deepConverting(dataFrame, column)
            
        else:
            # Convert the column to numeric values (integers or floats)
            dataFrame[column] = dataFrame[column].apply(self.convert_to_number)
            
        return dataFrame

    def is_float(self, value):
        """Helper function to check if a value can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def linearDataTypes(self, dataFrame, filterData = True):
        """Apply column conversions across all columns."""
        for column in dataFrame.columns:  # Iterate over each column in the DataFrame
            dataFrame = self.convert_column(dataFrame, column) 
        # Apply the conversion to each column
        # dataFrame = self.processor.automateRemovingNullValues()
        if (filterData):
            nullC = NullCheck(dataFrame)
            dataFrame = nullC.automateRemovingNullValues()
            
        return dataFrame  # Return the updated DataFrame





class Duplicated:
    def __init__(self):
        pass

    def getDuplicates(self, df):
        duplicateValues = {}  # Dictionary to hold duplicated values and their corresponding indexes
        # Loop through each column to find duplicate values
        for column in df.columns:
            duplicateValues[column] = self.getColumnDuplicates(df, column)
        return duplicateValues

    def getColumnDuplicates(self, df, column):
        # Find duplicated values in the column (including the first occurrence)
        duplicates = df[df[column].duplicated(keep=False)]  # Keep all duplicates (including the first)
        # Create a dictionary to hold the values and their corresponding indexes
        value_indexes = {}
        
        for idx, value in duplicates[column].items():
            if value not in value_indexes:
                value_indexes[value] = []  # Create a list for each unique duplicated value
            value_indexes[value].append(idx)  # Append the index to the list
        
        return value_indexes
    def removeDuplicates(self, df):
        df_cleaned = df[~df.duplicated(keep = "first")]
        return df_cleaned
    def replaceDuplicates(self, df):
        for column in df.columns:
            df = self.replaceColumnDuplicates(df, column)
        return df

    def replaceWithMode(self, df, column, duplicateList):
        """Replace duplicates in the column with the mode."""
        mode_value = df[column].mode()[0]  # Get the most frequent value (mode)
        # Replace all the duplicates with the mode
        for value, indices in duplicateList.items():
            for idx in indices[1:]:  # Skip the first occurrence (keep it)
                df.at[idx, column] = mode_value
        return df

    def replaceWithMedian(self, df, column, duplicateList):
        """Replace duplicates in the column with the median."""
        median_value = df[column].median()  # Get the median value
        # Replace duplicates with the median value
        for value, indices in duplicateList.items():
            for idx in indices[1:]:
                df.at[idx, column] = median_value
        return df

    def replaceWithMean(self, df, column, duplicateList):
        """Replace duplicates in the column with the mean."""
        mean_value = df[column].mean()  # Get the mean value
        # Replace duplicates with the mean value
        for value, indices in duplicateList.items():
            for idx in indices[1:]:
                df.at[idx, column] = mean_value
        return df

    def replaceColumnDuplicates(self, df, column):
        # getting the list of duplicated values except first occurence
        duplicateList = self.getColumnDuplicates(df, column)
        linear = FixDataTypes()
        linearData = linear.linearDataTypes(df)
        nullC = NullCheck(linearData)
        
        if (isCategorical(df, column)):
            linearData = self.replaceWithMode(linearData, column, duplicateList)
        elif (pd.api.types.is_numeric_dtype(linearData[column])):
            if (nullC.skewCheck(linearData[column])):
                linearData = self.replaceWithMedian(linearData, column, duplicateList)
            else:
                linearData = self.replaceWithMean(linearData, column, duplicateList)
        else:
            linearData = self.replaceWithMode(linearData, column, duplicateList)
        
        return linearData
                   


                   


class NormalityTest:
    def __init__(self, df):
        self.df = df
        
    def check_p_value(self, p_value):
        """
        Helper function to return True or False based on the p-value.
        """
        return p_value > 0.05  # Normal if p > 0.05
    
    def shapiro_wilk_test(self, data):
        """
        Shapiro-Wilk Test for normality
        """
        stat, p_value = stats.shapiro(data)
        return self.check_p_value(p_value), p_value

    def dagostino_pearson_test(self, data):
        """
        D'Agostino and Pearson's Test for normality
        """
        stat, p_value = stats.normaltest(data)
        return self.check_p_value(p_value), p_value

    def kolmogorov_smirnov_test(self, data):
        """
        Kolmogorov-Smirnov Test for normality (comparing against normal distribution)
        """
        stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        return self.check_p_value(p_value), p_value
    
    def jarque_bera_test(self, data):
        """
        Jarque-Bera Test for normality (based on skewness and kurtosis)
        """
        stat, p_value = stats.jarque_bera(data)
        return self.check_p_value(p_value), p_value

    def check_normality(self):
        """
        Main method to check normality for all numeric columns in the DataFrame.
        """
        normality_results = {}
        
        # Loop over each numerical column in the DataFrame
        for column in self.df.select_dtypes(include=[np.number]).columns:
            column_data = self.df[column].dropna()  # Drop missing values

            if len(column_data) < 3:  # Skip columns with fewer than 3 data points
                normality_results[column] = False
                continue
            
            # Perform the normality tests
            tests = [
                ("Shapiro-Wilk Test", self.shapiro_wilk_test),
                ("D'Agostino and Pearson's Test", self.dagostino_pearson_test),
                ("Kolmogorov-Smirnov Test", self.kolmogorov_smirnov_test),
                ("Jarque-Bera Test", self.jarque_bera_test)
            ]
            
            for test_name, test_func in tests:
                is_normal, p_value = test_func(column_data)
                if not is_normal:
                    normality_results[column] = False
                    break
            else:
                normality_results[column] = True  # If all tests passed, it's normal
        
        return normality_results

class DataCheck:
    def __init__(self):
        pass
    def check_high_variation(self, df, column, variance_threshold=0.1):
        if column not in df.columns:
            print(f"Column '{column}' not found in the DataFrame.")
            return False
        column_data = df[column]
        if pd.api.types.is_numeric_dtype(column_data):
            std_dev = column_data.std()
            if std_dev > variance_threshold:
                return True
            else:
                return False
        else:
            return False
        
    def checkVariablity(self, df, column, threshold=0.1):
        if (column in usefullColumns):
            return False
    # Check if the column is numeric
        column = df[column]
        if pd.api.types.is_numeric_dtype(column):
            # Check for binary values (0 and 1)
            if set(column.unique()).issubset({0, 1}):
                return False
        
            variance = column.var()
            uniqueCount = len(column.unique())
            rangeValue = column.max() - column.min()
        
            if variance < threshold and uniqueCount / len(column) < threshold:
                return True
        return False

    def checkHighCardinality(self, column, threshold = 0.1):
        uniqueCount = column.nunique()
        totalCount = len(column)
        ratio = uniqueCount / totalCount
        return ratio > threshold

    def checkCategoricalCardinality(self, column, thresholds=None):
        if column.empty:
            raise ValueError("The input column is empty.")

        unique_count = column.nunique()
        total_count = len(column)

        if thresholds is None:
            thresholds = {
                "low": 0.7,
                "medium": 0.3,
                "high": 0.2,
                "default": 0.3
            }

        ratio = unique_count / total_count

        if total_count < 200:
            return ratio <= thresholds["low"]
        elif 200 <= total_count <= 1000:
            return ratio <= thresholds["default"]
        elif 1000 < total_count < 10000:
            return ratio > thresholds["medium"]
        else:
            return ratio > thresholds["high"]
        

class Plot:
    def __init__(self):
        pass
    
    def create_plot(self, dataFrame, plotDescription, plotType='All'):
   
    
    # Set soothing color palette for plots
        sns.set_palette("muted")
    
    # Adjust figure size for small plots
        plt.figure(figsize=(6, 4))
        if (plotDescription == "MixData"):
            fd = FixDataTypes()
            fd.showDuplicates(dataFrame)
            return
            

        elif plotDescription == "Outliers":
        # Outliers: Detect outliers using different plot types
            if plotType == 'boxplot':
            # Boxplot to detect outliers in the 'Value' column
                sns.boxplot(x=dataFrame['Value'])
                plt.title("Outliers - Boxplot for 'Value' Column")
                plt.ylabel('Value')

            elif plotType == 'scatter':
            # Scatter plot to show outliers
                plt.scatter(dataFrame.index, dataFrame['Value'], alpha=0.6, color='orange')
                plt.title("Outliers - Scatter plot for 'Value' Column")
                plt.xlabel('Index')
                plt.ylabel('Value')

            elif plotType == 'violin':
            # Violin plot to show outliers and distribution
                sns.violinplot(x=dataFrame['Value'])
                plt.title("Outliers - Violin plot for 'Value' Column")
                plt.ylabel('Value')
    
        elif plotDescription == "DuplicateData":
        # DuplicateData: Show duplicate information using different plot types
            if plotType == 'bar':
            # Bar plot for duplicate counts
                duplicate_count = dataFrame.duplicated().sum()
                unique_count = len(dataFrame) - duplicate_count
                plt.bar(['Duplicates', 'Unique'], [duplicate_count, unique_count], color=['lightcoral', 'lightgreen'])
                plt.title("Duplicate Data - Count of Duplicates vs Unique")
                plt.ylabel('Count')

            elif plotType == 'pie':
            # Pie chart to show proportion of duplicates vs unique
                unique_values = dataFrame.duplicated().sum()
                total_values = len(dataFrame)
                plt.pie([unique_values, total_values - unique_values], labels=["Duplicates", "Unique"], autopct='%1.1f%%', startangle=90)
                plt.title("Duplicate Data - Proportion of Duplicates vs Unique")

            elif plotType == 'heatmap':
            # Heatmap to visualize duplicates
                duplicate_matrix = dataFrame.duplicated(keep=False).astype(int)
                sns.heatmap(duplicate_matrix.values.reshape(-1, 1), cmap='Blues', cbar=False)
                plt.title("Duplicate Data - Heatmap of Duplicates")

        elif plotDescription == "NonLinearData":
        # NonLinearData: Visualize non-linear relationships
            if plotType == 'scatter':
            # Scatter plot with potential non-linear data
                sns.scatterplot(x=dataFrame['Value'], y=dataFrame['Score'], color='skyblue')
                plt.title("Non-linear Data - Scatterplot of 'Value' vs 'Score'")
                plt.xlabel('Value')
                plt.ylabel('Score')

            elif plotType == 'line':
            # Line plot (non-linear regression line)
                sns.regplot(x=dataFrame['Value'], y=dataFrame['Score'], scatter_kws={'color': 'skyblue'}, line_kws={'color': 'orange'})
                plt.title("Non-linear Data - Line plot with Regression")
                plt.xlabel('Value')
                plt.ylabel('Score')

        elif plotDescription == "All":
        # Default plot: Histogram of 'Value' column (you can modify this to any default behavior you want)
            sns.histplot(dataFrame['Value'], kde=True, bins=20, color='lightgreen')
            plt.title("All Data - Histogram of 'Value' Column")
            plt.xlabel('Value')
            plt.ylabel('Frequency')

        else:
            print(f"Plot description '{plotDescription}' is not recognized.")
    
    # Display the plot
        plt.tight_layout()
        plt.show()
    
    
class PreprocessData:
    def __init__(self):
        pass

    def preprocessData(self, dataFrame, targetColumn,nullR = True, treatOutlier = False, showOutliers = False):
        nullC = NullCheck(dataFrame)
        dup = Duplicated()
        fDtype = FixDataTypes()
        # norm = NormalityTest(df)
        df = fDtype.linearDataTypes(dataFrame, nullR)
        # adf = dup.replaceDuplicates(df)
        outlierD = OutlierDetection(df, targetColumn)
        
        if (treatOutlier):
            df = outlierD.automateOutliers()
        if (showOutliers):
            outlierD.showOutliers()
        
        return df
        
        
        
