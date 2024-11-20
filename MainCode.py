import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import random
import string
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

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
                sns.set_palette(["#FFA07A"])  # Light Salmon (light orange)

                sns.set_style("whitegrid")  # Adds a soft white grid background

                plt.figure(figsize=(5, 3))

                if plot_type == "boxplot":
                    sns.boxplot(y=self.df[column], color="#FF8C00")  # Skyblue for a calming look
                    plt.title(f"Box Plot of {column}", fontsize=14, fontweight='bold')
                    plt.ylabel(column, fontsize=12)  # Fixed the typo here
                elif plot_type == 'scatter':
                    plt.scatter(self.df.index, self.df[column], color='#FF8C00', alpha=0.7)  # Light coral for soothing color
                    plt.title(f'Scatter Plot of {column}', fontsize=14, fontweight='bold')
                    plt.ylabel(column, fontsize=12)
                    plt.xlabel('Index', fontsize=12)
                elif plot_type == 'histogram':
                    sns.histplot(self.df[column], bins=30, kde=True, color='#FF8C00')  # Lightseagreen for a calm histogram color
                    plt.title(f'Histogram of {column}', fontsize=14, fontweight='bold')
                    plt.xlabel(column, fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
        
# Adjust font sizes for readability
                plt.tight_layout()
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

    
    def detectOutliers(self, count = True):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            # Skip non-numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()

                # Skip columns with constant values (no variance)
                if data.nunique() <= 2:
                    continue
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
        
    def detectColumnOutliers(self, column, boolean=False):
        outliers = {}

        # Check if the specified column is numeric
        if pd.api.types.is_numeric_dtype(self.df[column]):
            columnSkewness = self.df[column].skew()
        
            # If skewness is high, use the IQR method
            if abs(columnSkewness) > 0.5:
                outliers[column] = self.iqrOutliers(self.df[column])
            else:
                # If skewness is low, use the standard deviation method
                outliers[column] = self.sdOutliers(self.df[column])

            # If `boolean` is True, return True if outliers exist, otherwise False
            if boolean:
                return len(outliers[column]) > 0  # True if outliers are detected, False otherwise

        elif pd.api.types.is_object_dtype(self.df[column]):
            # If the column is categorical, use a categorical outlier detection method
            outliers[column] = self.detectCategoricalOutliers(column)
        
            # If `boolean` is True, return True if outliers exist, otherwise False
            print(outliers[column])
            if boolean:
                return len(outliers[column]) > 0  # True if outliers are detected, False otherwise

        # If `boolean` is False, return the outliers dictionary for that column
            
        return outliers

    def detectDynamicOutliers(self, boolean=False):
        allOutliers = {}

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
            targetSkewness = self.df[self.target_column].skew()
            if abs(targetSkewness) > 0.5:
                # Use IQR for skewed data
                targetOutliers = self.iqrOutliers(self.df[self.target_column])
            else:
                # Use SD method for data that's not skewed
                targetOutliers = self.sdOutliers(self.df[self.target_column])

            if len(targetOutliers) != 0:
                allOutliers[self.target_column] = targetOutliers
        else:
            # Categorical data outlier detection
            targetOutliers = self.detectCategoricalOutliers(self.target_column)
            if len(targetOutliers) != 0:
                allOutliers[self.target_column] = targetOutliers

        # Iterate through all columns except the target column
        for column in self.df.columns:
            if column == self.target_column:
                continue
        
            # For numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                columnSkewness = self.df[column].skew()
                if abs(columnSkewness) > 0.5:
                    outliers = self.iqrOutliers(self.df[column])  # Skewed data -> IQR
                else:
                    outliers = self.sdOutliers(self.df[column])  # Normal data -> SD method
            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, use a specific method
                outliers = self.detectCategoricalOutliers(column)
        
            # Only add columns with detected outliers
            if len(outliers) != 0:
                allOutliers[column] = outliers

        # If `boolean` is True, return True if any outliers were detected, otherwise False
        if boolean:
            return len(allOutliers) > 0  # Return True if there are any outliers, otherwise False

        # If `boolean` is False, return the dictionary of all detected outliers
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
        allOutliers = self.detectOutliers(count = False)
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
   


    def replaceDuplicates(self, df):
        """Replace duplicates in all columns with appropriate values."""
        for column in df.columns:
            df = self.replaceColumnDuplicates(df, column)
        return df

    def replaceWithMean(self, df, column, duplicateIndices):
        """Replace duplicates in the column with the mean."""
        mean_value = df[column].mean()  # Get the mean value
    
    # Iterate over the duplicate values and their indices
        for value, indices in duplicateIndices.items():
        # Ensure indices is a list (in case it's not)
            if not isinstance(indices, list):
                indices = [indices]  # Convert to list if it's not already
        
            # Skip the first occurrence (keep it), and replace the rest with mean
            for idx in indices[1:]:
                df.at[idx, column] = mean_value  # Use .at to set the value at a specific index
    
        return df



    def replaceColumnDuplicates(self, df, column):
        """Identify duplicates in the column and replace them based on their data type."""
        duplicateList = self.getColumnDuplicates(df, column)  # Get duplicate values and their indices
        
        # Handle categorical columns (use Mode)
        if self.isCategorical(df, column):
            return self.replaceWithMode(df, column, duplicateList)
        
        # Handle numerical columns (use Median or Mean depending on skew)
        if pd.api.types.is_numeric_dtype(df[column]):
            if self.isSkewed(df[column]):
                return self.replaceWithMedian(df, column, duplicateList)
            else:
                return self.replaceWithMean(df, column, duplicateList)

        # Default: For non-numeric, non-categorical (like dates, text), use Mode
        return self.replaceWithMode(df, column, duplicateList)

    def getColumnDuplicates(self, df, column):
        """Get duplicate indices for each value in the column."""
        # Find all duplicates, keeping all occurrences (not just the first)
        duplicates = df[df[column].duplicated(keep=False)]
        # Group the duplicates by their value, and return the indices in a list
        duplicate_indices = duplicates.groupby(column).apply(lambda x: x.index.tolist()).to_dict()
        return duplicate_indices


    def isCategorical(self, df, column):
        """Check if the column is categorical."""
        return df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column])

    def isSkewed(self, series):
        """Check if the data is skewed using skewness metric."""
        return series.skew() > 1  # A simple rule of thumb for positive skewness

    # Additional methods for replacing with Mode and Median, not shown for brevity
    def replaceWithMode(self, df, column, duplicateIndices):
        """Replace duplicates in the column with the mode."""
        mode_value = df[column].mode()[0]  # Get the most frequent value (mode)
    
        for value, indices in duplicateIndices.items():
            if isinstance(indices, list) and len(indices) > 1:  # Check if indices are a list and have more than one duplicate
                # Skip the first occurrence (keep it), and replace the rest with mode
                for idx in indices[1:]:
                    df.at[idx, column] = mode_value  # Use .at to set the value at a specific index
        return df


    def replaceWithMedian(self, df, column, duplicateIndices):
        """Replace duplicates in the column with the median."""
        median_value = df[column].median()  # Get the median value
        # Iterate over the duplicate values and their indices
        for value, indices in duplicateIndices.items():
            if isinstance(indices, list) and len(indices) > 1:  # Check if indices are a list and have more than one duplicate
            
            # Skip the first occurrence (keep it), and replace the rest with median
                for idx in indices[1:]:
                    df.at[idx, column] = median_value  # Use .at to set the value at a specific index
        return df
    def plotRowDuplicatesBarChart(self, df):
        """Generate a bar chart showing duplicates vs unique values for each row."""
        
        # Initialize lists to hold the counts for each row
        duplicate_counts = []
        unique_counts = []

        # Iterate over each row to count duplicates and unique values
        for idx, row in df.iterrows():
            value_counts = row.value_counts()  # Count how many times each value appears in the row
            
            # Count duplicates (values that appear more than once)
            duplicates = sum(value_counts > 1)
            unique = len(value_counts) - duplicates  # Unique values are those that appear exactly once
            
            duplicate_counts.append(duplicates)
            unique_counts.append(unique)
        
        # Create a DataFrame to hold the counts for easy plotting
        counts_df = pd.DataFrame({
            'Duplicates': duplicate_counts,
            'Unique': unique_counts
        })

        # Plot the bar chart
        counts_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightblue', 'lightgreen'])
        plt.title('Duplicates vs Unique Values Per Row')
        plt.xlabel('Row Index')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Count Type')
        plt.tight_layout()
        plt.show()

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
        
    
    
 
class IrrelvantColumns:
    def __init__(self):
        pass

    def constantValue(self, column):
        return column.nunique() == 1

    def check_high_cardinality_low_frequency(self, df, column, cardinality_threshold=0.1, frequency_threshold=0.05):
        # Calculate the number of distinct values
        num_distinct_values = df[column].nunique()
        num_rows = len(df)
    
        # High cardinality check: More distinct values than the threshold percentage of total rows
        if num_distinct_values / num_rows < cardinality_threshold:
            return False
    
        # Check frequency of values
        value_counts = df[column].value_counts(normalize=True)
    
        # Check if a significant portion of the values have a low frequency (below the threshold)
        low_frequency_count = sum(value_counts[value_counts < frequency_threshold])
    
        # High cardinality and low frequency condition
        if low_frequency_count > 0.5:  # At least 50% of the distinct values are low frequency
            return True
    
        return False

    def is_highly_skewed(self, df, column, threshold=1.0):
        # Check if the column exists in the DataFrame
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
        # Ensure the column is numerical
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not a numerical column.")
    
        # Calculate skewness of the column, dropping any NaN values
        skewness_value = df[column].skew()  # .dropna() handles missing values
    
        # Return True if the absolute skewness is greater than the threshold, otherwise False
        return abs(skewness_value) > threshold

    def find_identical_columns_optimized(self, df):
        identical_column_pairs = []
        column_hashes = {}

        for col in df.columns:
            column_hash = hash(tuple(df[col].values))
        
            if column_hash in column_hashes:
                identical_column_pairs.append((column_hashes[column_hash], col))
            else:
                column_hashes[column_hash] = col

        return identical_column_pairs

    def check_sparse_data(self, df, column, threshold=0.9):
        """Check if a column has too many unique values compared to the total number of rows."""
        num_distinct_values = df[column].nunique()
        num_rows = len(df)
    
        # If the proportion of unique values exceeds the threshold, flag as sparse
        if num_distinct_values / num_rows > threshold:
            return True
    
        return False

    def removeColumns(self, df, targetColumn, threshold=1.0, cardinality_threshold=0.1, frequency_threshold=0.05, sparse_threshold=0.9):
        removalList = {
            'constant_values': [],
            'high_cardinality_low_frequency': [],
            'highly_skewed': [],
            'useless_columns': [],
            'identical_columns': [],
            'sparse_columns': [],  # Added for sparse columns
            "Outliers": []
        }

        # Identify identical columns first
        removalList['identical_columns'] = self.find_identical_columns_optimized(df)

        # Assuming OutlierDetection is defined elsewhere in your code
        ot = OutlierDetection(df, targetColumn)
        oList = ot.detectOutliers()
        removalList["Outliers"].append(oList)

        # Loop through each column and classify it based on the criteria
        for column in df.columns:
            if self.constantValue(df[column]):
                removalList['constant_values'].append(column)

            if self.check_high_cardinality_low_frequency(df, column, cardinality_threshold, frequency_threshold):
                removalList['high_cardinality_low_frequency'].append(column)

            if pd.api.types.is_numeric_dtype(df[column]):
                if self.is_highly_skewed(df, column, threshold):
                    removalList['highly_skewed'].append(column)

            # Check for sparse columns
            if self.check_sparse_data(df, column, sparse_threshold):
                removalList['sparse_columns'].append(column)

            # Assuming `unique_identifiers` is defined elsewhere, and its logic is correct
            if column in unique_identifiers:
                removalList['useless_columns'].append(column)

        return removalList
  
    
        





class Plot:
    def __init__(self):
        pass
    
    def showPlot(self, dataFrame,targetColumn, plotDescription, plotType):
        if (plotDescription == "null"):
            nullC = NullCheck(dataFrame)
            nullC.visualizeMissingData()
        elif (plotDescription == "outliers"):
            od = OutlierDetection(dataFrame, targetColumn)
            
            if (plotType == "scatter"):
                od.showOutliers(plot_type = "scatter")
            elif (plotType == "histogram"):
                od.showOutliers(plot_type = "histogram")
            else:
                od.showOutliers()
                
        elif (plotDescription == "linearity"):
            fd = FixDataTypes()
            fd.showDuplicates(dataFrame)
        elif (plotDescription == "duplicates"):
            dup = Duplicated()
            dup.plotRowDuplicatesBarChart(dataFrame)
        else:
            print("Working")
            
        
            
                    
                    
                    
                    
        
            
    
class Normalization:
    def __init__(self):
        pass
        
    def minMax(self, df, column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)
        return df

    def zScore(self, df, column):
       
        mean = df[column].mean()
        std = df[column].std()

        df[column] = (df[column] - mean) / std
        return df

    def robustScaling(self, df, column):
       
        scaler = RobustScaler()

        df[column] = scaler.fit_transform(df[[column]])
        return df

    def logTransformation(self, df, column):
    
        df[column] = np.log1p(df[column])  # log(x + 1)
        return df

    def decimalScaling(self, df, column):
      
        max_abs_value = df[column].abs().max()
        scaling_factor = 10 ** np.ceil(np.log10(max_abs_value))

        df[column] = df[column] / scaling_factor
        return df

    def unitVector(self, df, column):
        
        norm = np.linalg.norm(df[column])

        if norm != 0:
            df[column] = df[column] / nrm
        return df
    def sd_based_outlier_detection(self, df, column, threshold=3):
        mean = df[column].mean()
        std_dev = df[column].std()
        upper_bound = mean + threshold * std_dev
        lower_bound = mean - threshold * std_dev

        outliers = (df[column] > upper_bound) | (df[column] < lower_bound)
        return outliers
    
    # def automateNormalization(self, df):
    #     # Check for and remove null values
    #     nullC = NullCheck(df)
    #     df = nullC.automateRemovingNullValues()

    #     # Identify numeric columns
    #     numeric_columns = df.select_dtypes(include=[np.number]).columns
    #     if len(numeric_columns) == 0:
    #         raise ValueError("No numeric columns found for normalization.")
    #     if df.empty:
    #         raise ValueError("The DataFrame is empty.")

    #     # Iterate through each numeric column for normalization
    #     for column in numeric_columns:
    #         print(f"Processing column: {column}")

    #         # Detect outliers using the SD method
    #         outliers = self.sd_based_outlier_detection(df, column)
    
    #         # Apply different normalization techniques based on conditions
    #         if outliers.any():  # If there are any outliers detected
    #             print(f"Outliers detected in {column}. Applying Robust Scaling.")
    #             df = self.robustScaling(df, column)

    #         elif df[column].std() != 0 and abs(df[column].skew()) <= 0.5:
    #             # Apply Z-Score Normalization for nearly normal distributions
    #             print(f"{column} is approximately normal. Applying Z-Score Normalization.")
    #             df = self.zScore(df, column)

    #         elif df[column].skew() > 0.8:  # Adjusted threshold for heavily skewed columns
    #         # Apply Log Transformation for heavily skewed, positive values
    #             if (df[column] > 0).all():
    #                 print(f"{column} is heavily skewed. Applying Log Transformation.")
    #                 df = self.logTransformation(df, column)
    #             else:
    #                 print(f"Skipping Log Transformation for {column} due to non-positive values.")

    #         elif df[column].min() >= 0 and df[column].max() <= 100:
    #             # Apply Min-Max Scaling if values are between 0 and 100
    #             print(f"Applying Min-Max Scaling to {column}.")
    #             df = self.minMax(df, column)

    #         else:
    #             # Apply Decimal Scaling as a fallback for non-skewed, non-outlying columns
    #             print(f"Applying Decimal Scaling to {column}.")
    #             df = self.decimalScaling(df, column)

    #     # Return the modified DataFrame after applying appropriate normalization techniques
    #     return df


    def getAllNormality(self, df):
        # Initialize an empty dictionary to store the results
        columnList = {}

        # Loop through each column in the DataFrame
        for column in df.columns:
            # Ensure the column contains numeric data
            if pd.api.types.is_numeric_dtype(df[column]):
                # Call the getNormality function for each column
                normality_results = self.getNormality(df, column)
                # Add the normality results to the dictionary
                columnList[column] = normality_results
            else:
                # For non-numeric columns, we can skip or handle differently
                columnList[column] = {'normality_score': 'N/A', 'skewness': 'N/A', 'kurtosis': 'N/A', 'shapiro_p_value': 'N/A'}

        # Return the dictionary containing the normality information for all columns
        normality_df = pd.DataFrame(columnList).T  # Transpose to get columns as rows
        normality_df = normality_df.reset_index()  # Reset index for better readability
        normality_df.rename(columns={'index': 'Column'}, inplace=True)  # Rename index column to 'Column'

        # Return the DataFrame for better readability
        return normality_df
    def showNormalityGraphs(self, df):
        """Display histograms and Q-Q plots for each numeric column in the DataFrame."""
        # Determine number of numeric columns
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        num_columns = len(numeric_columns)

        # Calculate grid size (number of rows and columns) for subplots
        num_rows = int(np.ceil(num_columns / 3))  # Ensure enough rows to fit all columns (3 columns per row)
        num_cols = 3  # Fix the number of columns to 3

        # Plotting setup for histograms
        plt.figure(figsize=(14, 5 * num_rows))  # Adjust figure height based on the number of rows

        # Loop through each numeric column to show the normality plots (Histograms)
        for idx, column in enumerate(numeric_columns):
            # Create subplots for each column (histogram)
            plt.subplot(num_rows, num_cols, idx + 1)
            
            # Plot Histogram with KDE (Kernel Density Estimate)
            sns.histplot(df[column], kde=True, bins=20)
            plt.title(f"Histogram for {column}")

        # Show all histograms
        plt.tight_layout()
        plt.show()

        # Plotting setup for Q-Q plots
        plt.figure(figsize=(14, 5 * num_rows))  # Adjust figure height for Q-Q plots

        # Loop through each numeric column to show the Q-Q plot
        for idx, column in enumerate(numeric_columns):
            plt.subplot(num_rows, num_cols, idx + 1)
            stats.probplot(df[column], dist="norm", plot=plt)
            plt.title(f"Q-Q plot for {column}")
        
        # Show all Q-Q plots
        plt.tight_layout()
        plt.show()

    def showNormality(self, df):
        normality_df = self.getAllNormality(df)

        # Determine number of numeric columns
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        num_columns = len(numeric_columns)

        # Calculate grid size (number of rows and columns) for subplots
        num_rows = int(np.ceil(num_columns / 3))  # Ensure enough rows to fit all columns (3 columns per row)
        num_cols = 3  # Fix the number of columns to 3

        # Plotting setup
        plt.figure(figsize=(14, 5 * num_rows))  # Adjust figure height based on the number of rows

        # Loop through each numeric column to show the normality plots
        for idx, column in enumerate(numeric_columns):
            # Create subplots for each column
            plt.subplot(num_rows, num_cols, idx + 1)
        
            # Plot Histogram with KDE
            sns.histplot(df[column], kde=True, bins=20)
            plt.title(f"Histogram for {column}")
        
            # Display Skewness, Kurtosis, and p-value on the plot
            normality_results = normality_df[normality_df['Column'] == column].iloc[0]
            skew = normality_results['skewness']
            kurt = normality_results['kurtosis']
            p_value = normality_results['shapiro_p_value']
            plt.xlabel(f"Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}, p-value: {p_value:.3f}")

        # Show all histograms
        plt.tight_layout()
        plt.show()

        # Q-Q plot for each numeric column to visually assess normality
        plt.figure(figsize=(14, 5 * num_rows))
        for idx, column in enumerate(numeric_columns):
            plt.subplot(num_rows, num_cols, idx + 1)
            stats.probplot(df[column], dist="norm", plot=plt)
            plt.title(f"Q-Q plot for {column}")
    
        # Show all Q-Q plots
        plt.tight_layout()
        plt.show()

        
        
    def getNormality(self, df, column):
        # Calculate skewness and kurtosis
        skewness = df[column].skew()
        kurtosis = df[column].kurtosis()
    
        # Perform the Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(df[column])
    
        # Now, determine how much the column deviates from normality
        normality_score = 0
    
        # Skewness: near 0 is ideal
        if abs(skewness) > 0.5:
            normality_score += abs(skewness)  # Penalize for skewness
    
        # Kurtosis: near 3 is ideal for a normal distribution
        if abs(kurtosis - 3) > 1:
            normality_score += abs(kurtosis - 3)  # Penalize for deviation from normal kurtosis
    
        # Shapiro-Wilk test: p-value > 0.05 means normal, less means non-normal
        if p_value < 0.05:
            normality_score += 1  # Increase score for non-normality
    
        # Return the normality score along with statistical results
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_p_value': p_value,
            'normality_score': normality_score
        }

    

    def manualNormalization(self, df, column, way=None):
        # Check if 'way' is provided
        if way is None:
            raise ValueError("Argument 'way' is required. Please specify the scaling method.")
    
        # Apply the appropriate transformation based on the 'way' argument
        if way == "minmax":
            print(f"Applying Min-Max Scaling to {column}")
            df = self.minMax(df, column)
        
        elif way == "zscore":
            print(f"Applying Z-Score Normalization to {column}")
            df = self.zScore(df, column)
        
        elif way == "robustscaling":
            print(f"Applying Robust Scaling to {column}")
            df = self.robustScaling(df, column)
        
        elif way == "logtransforming":
            print(f"Applying Log Transformation to {column}")
            df = self.logTransformation(df, column)
            
        elif way == "decimalScaling":
            print(f"Applying Decimal Scaling to {column}")
            df = self.decimalScaling(df, column)
        
        elif way == "unitvector":
            print(f"Applying Unit Vector Scaling to {column}")
            df = self.unitVector(df, column)
        
        else:
            raise ValueError(f"Invalid method '{way}' specified. Please choose from: 'minmax', 'zscore', 'robustscaling', 'logtransforming', 'decimalScaling', or 'unitvector'.")
    
        return df

        
            

# class InconData:
#     def __init__(self):
#         pass

#     # Check if column contains only boolean values or valid boolean-like strings
#     def checkBool(self, column):
#         valid_booleans = ['True', 'False', 'yes', 'no', '1', '0', 'y', 'n']
        
#         # Check if column is strictly boolean
#         if column.isin([True, False]).all():
#             return True
        
#         # Check if column contains only valid boolean-like strings
#         if column.astype(str).isin(valid_booleans).all():
#             return True
        
#         return False
    
#     # Check if column contains only numeric values (int, float)
#     def checkNumeric(self, column):
#         try:
#             pd.to_numeric(column, errors='raise')
#             return True
#         except ValueError:
#             return False
    
#     # Check if column is categorical or contains string values
#     def checkCategorical(self, column):
#         return column.dtype == 'object' or pd.api.types.is_categorical_dtype(column)
    
#     # Check if column can be converted to datetime
#     def checkDateTime(self, column):
#         try:
#             pd.to_datetime(column, errors='raise')
#             return True
#         except Exception:
#             return False
    
#     # Check if the length of the values in a column is consistent
#     def checkLength(self, column):
#         length = column.apply(lambda x: len(str(x))).mode().iloc[0]
#         if column.apply(lambda x: len(str(x)) != length).any():
#             return False
#         return True
    
#     # Consistency check for all columns in the DataFrame
#     def consistentData(self, df):
#         # Prepare results
#         results = []
        
#         for column in df.columns:
#             column_info = {'column': column}

#             # Check type of column and apply appropriate check
#             if self.checkBool(df[column]):
#                 column_info['type'] = 'bool'
#                 column_info['consistent'] = True
#             elif self.checkNumeric(df[column]):
#                 column_info['type'] = 'numeric'
#                 column_info['consistent'] = True
#             elif self.checkDateTime(df[column]):
#                 column_info['type'] = 'datetime'
#                 column_info['consistent'] = True
#             elif self.checkCategorical(df[column]):
#                 column_info['type'] = 'categorical'
#                 column_info['consistent'] = True
#             else:
#                 column_info['type'] = 'unknown'
#                 column_info['consistent'] = False
            
#             # Check for length consistency
#             column_info['length_consistent'] = self.checkLength(df[column])

#             results.append(column_info)
        
#         # Convert results into a DataFrame for better visualization
#         consistency_df = pd.DataFrame(results)
#         return consistency_df
   
    

class CategoricalEncoder:
    def __init__(self):
       pass

    def convert_categorical_column(self, df, column, encoding_type="onehot"):
       
        if encoding_type == "onehot":
            # One-Hot Encoding using pd.get_dummies
            return pd.get_dummies(df, columns=[column], prefix=[column])
        
        elif encoding_type == "label":
            # Label Encoding using LabelEncoder
            le = LabelEncoder()
            df[column + '_Label'] = le.fit_transform(df[column])
            return df
        
        else:
            raise ValueError("encoding_type must be 'onehot' or 'label'")


    def plot_categorical_distribution(self,df, column, encoding_type="onehot"):
        
        # Original distribution (before encoding)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        sns.countplot(data=df, x=column)
        plt.title(f"Original {column} Distribution")
        
        # Encoding the column
        if encoding_type == "label":
            # Perform Label Encoding
            df_encoded = self.convert_categorical_column(df, column, encoding_type="label")
            encoded_column = column + "_Label"
            plt.subplot(1, 3, 2)
            sns.countplot(data=df_encoded, x=encoded_column)
            plt.title(f"Label Encoded {column} Distribution")
        
        elif encoding_type == "onehot":
            # Perform One-Hot Encoding
            df_encoded = self.convert_categorical_column(df, column, encoding_type="onehot")
            onehot_columns = [col for col in df_encoded.columns if column in col]
            # Sum of each one-hot encoded column
            onehot_sums = df_encoded[onehot_columns].sum()
            plt.subplot(1, 3, 2)
            sns.barplot(x=onehot_sums.index, y=onehot_sums.values)
            plt.title(f"One-Hot Encoded {column} Distribution")

        plt.tight_layout()
        plt.show()

class PreprocessData:
    def __init__(self):
        pass
    def describeDifference(self, original_df, result_df):
        # Get describe() for both original and result DataFrames
        original_desc = original_df.describe().T  # Transpose for easy comparison
        result_desc = result_df.describe().T

        # Align the DataFrames (matching columns, filling missing with NaN)
        combined_desc = pd.concat([original_desc, result_desc], axis=1, keys=['Original', 'Result'])

        # Replace NaN with 'null' in case columns are missing
        combined_desc = combined_desc.fillna('null')

        # Calculate the difference between original and result
        # This creates a DataFrame where each statistic is subtracted row-wise
        difference = original_desc.subtract(result_desc, fill_value=0)
    
        # Now add the difference to combined_desc by concatenating along the columns axis
        # The difference DataFrame will also be transposed so that it's aligned correctly
        combined_desc = pd.concat([combined_desc, difference.rename(columns={'Original': 'Difference'})], axis=1)

        return combined_desc

    def preprocessDataFunctionUse(self, dataFrame):
        fDtype = FixDataTypes()
        df = fDtype.linearDataTypes(dataFrame, True)
        return df
        
        
        
        
    def preprocessData(self, dataFrame, targetColumn, nullR = True, treatOutlier = False, showOutliers = False, replace = False):
        print(dataFrame.describe())
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
        if (replace):
            df = dup.replaceDuplicates(df)
        # diff_df = self.describeDifference(dataFrame, df)
        # print("Differences between original and resulting DataFrame:")
        # print(diff_df)
        print(df.describe())
        irrelvantColumnList = IrrelvantColumns()
        iList = irrelvantColumnList.removeColumns(df, targetColumn)
        print(iList)
        
        return df
        
        
        
    
    
        



