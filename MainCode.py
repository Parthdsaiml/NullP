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
        self.outlier_detection = OutlierDetection(df)  # Initialize OutlierDetection class

        
    
    # Null Check: Check for missing values and display summary
    def nullCheck(self):
        totalNullsCombineColumns = self.df.isnull().sum().sum()
        if totalNullsCombineColumns != 0:
            print("Total Nulls: ", totalNullsCombineColumns)
            print("Null Count in Each Column:")
            print(self.df.isnull().sum())
        print("No Null Values found")
        return None
    def automateRemovingNullValues(self, threshold=0.1):
        for column in self.df.columns:
            # Skip large columns (more than 1 million entries)
            if len(self.df[column]) > 1000000:
                continue  # Need a better approach to handle very large columns
        
        # Handle categorical columns (replace NaNs with Mode)
            if isCategorical(self.df, column):
                self.replaceWithMode(column)
            else:
            # Check if the column is numeric (not categorical)
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    if self.skewCheck(self.df[column]):
                    # If the column is skewed, check for outliers using IQR or SD
                        iqr_outliers = self.outlier_detection.iqrOutliers(self.df[column])
                        sd_outliers = self.outlier_detection.sdOutliers(self.df[column])

                    # If outliers are detected, use Mode to replace missing values
                        if iqr_outliers or sd_outliers:
                            self.replaceWithMode(column)
                        else:
                        # Replace with Median for skewed columns without outliers
                            self.replaceWithMedian(column)
                    else:
                    # For non-skewed numeric columns, replace with Mean
                        self.replaceWithMean(column)
                else:
                # Handle non-numeric columns
                    self.replaceWithMode(column)

    def skewCheck(self, column, skew_threshold=0.5, kurtosis_threshold=3.0):
        if not pd.api.types.is_numeric_dtype(column):
            return False
        skewness = column.skew()
        kurt = column.kurtosis()
        skewed = abs(skewness) > skew_threshold
        heavy_tailed = abs(kurt) > kurtosis_threshold
        if skewed or heavy_tailed:
            return True
        else:
            return False

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

    # Replace missing values with Median (for numeric data)
    def replaceWithMedian(self, column):
        medianValue = self.df[column].median()
        self.df[column] = self.df[column].fillna(medianValue)
        return self.df

    # KNN Imputation Method for replacing missing values
    def knnMethodNullReplacement(self, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(self.df)
        imputed_df = pd.DataFrame(imputed_data, columns=self.df.columns)
        return imputed_df

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
    

    def detectOutliers(self, count=False):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                continue

            # Drop missing values for outlier detection
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

            # Identify the outliers in the column
            outliers = data[(data < lower_bound) | (data > upper_bound)]

            # If outliers exist, store either the count or the list of outliers
            if not outliers.empty:
                if count:
                    all_outliers[column] = len(outliers)  # Store count of outliers
                else:
                    all_outliers[column] = outliers.tolist()  # Store list of outliers

        return all_outliers


   






        
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
                series[series.isin(outliers)] = median_value

            else:
                print("Invalid method specified. Please use 'remove', 'cap', or 'impute'.")
        return series
        
    def automateOutliers(self):
        allOutliers = self.detectAllOutliers()
        if self.target_column in allOutliers:
            targetOutliers = allOutliers[self.target_column]
            self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers)

        for column in self.df.columns:
            columnOutliers = allOutliers[column]
            self.df[column] = self.handleOutliers(self.df[column], columnOutliers)
            
        return self.df
      
    def automateOutliersAndNormalisation(self, target = False, columnC = False):
        allOutliers = self.detectAllOutliers()
        
        # Step 1: Handle outliers in the target column
        if self.target_column in allOutliers:
            targetOutliers = allOutliers[self.target_column]
            self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers)

        # Step 2: Handle outliers in other columns
        for column in self.df.columns:
            if column != self.target_column and column in allOutliers:
                columnOutliers = allOutliers[column]
                self.df[column] = self.handleOutliers(self.df[column], columnOutliers)

        # Step 3: Apply transformations to normalize the data
        if (target and columnC):
            self.apply_transformation()
        elif (target):
            self.apply_transformationJustTarget()
        elif (columnC):
            self.apply_transformation()
        else:
            self.apply_transformation()
    
     
    def apply_transformationJustTarget(self):
        data = self.df[self.target_column]  # Get the target column from the DataFrame
        skewness = stats.skew(data)
        
        # Step 1: Check for negative or zero values and apply Yeo-Johnson if needed
        if np.any(data <= 0):
            pt = PowerTransformer(method='yeo-johnson')
            self.df[self.target_column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()  # Apply Yeo-Johnson

        # After Yeo-Johnson, re-check skewness
        data = self.df[self.target_column]  # Re-get the target column after transformation
        skewness = stats.skew(data)
        
        # Step 2: Apply transformations based on skewness
        if skewness > 1:  # Positively skewed data
            self.df[self.target_column] = np.log(data + 1)  # Log transformation (adding 1 to handle zero values)
        elif skewness < -1:  # Negatively skewed data
            # Box-Cox requires strictly positive values
            self.df[self.target_column], _ = stats.boxcox(data[data > 0])  # Filter out non-positive values for Box-Cox
        elif 0 < skewness <= 1:  # Moderately positively skewed data
            self.df[self.target_column] = np.sqrt(data)  # Square root transformation
        elif -1 <= skewness < 0:  # Moderately negatively skewed data
            # Box-Cox transformation for moderately skewed negative data (requires positive values)
            self.df[self.target_column], _ = stats.boxcox(data[data > 0])
        # After transformation, re-check normality
        return self.check_normality()
        
    def applyTransofmation(slef):
        for column in self.df.columns:
            data = self.df[column]
            skewness = stats.skew(data)
            # Apply Yeo-Johnson if there are negative or zero values
            if np.any(data <= 0):
                pt = PowerTransformer(method='yeo-johnson')
                self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
            # After transformation, re-check skewness and apply appropriate transformations
            data = self.df[column]
            skewness = stats.skew(data)
            # Apply transformations based on skewness
            if skewness > 1:  # Positively skewed data
                self.df[column] = np.log(data + 1)
            elif skewness < -1:  # Negatively skewed data
                self.df[column], _ = stats.boxcox(data[data > 0])
            elif 0 < skewness <= 1:  # Moderately positively skewed
                self.df[column] = np.sqrt(data)
            elif -1 <= skewness < 0:  # Moderately negatively skewed
                self.df[column], _ = stats.boxcox(data[data > 0])
        return self.check_normality()
        
    def applyTransformationExceptTarget(self):
    # Apply transformations to all columns, except the target column
        for column in self.df.columns:
            if column != self.target_column:  # Skip the target column
                data = self.df[column]
                skewness = stats.skew(data)

            # Apply Yeo-Johnson if there are negative or zero values
                if np.any(data <= 0):
                    pt = PowerTransformer(method='yeo-johnson')
                    self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
            # After transformation, re-check skewness and apply appropriate transformations
                data = self.df[column]
                skewness = stats.skew(data)
            # Apply transformations based on skewness
                if skewness > 1:  # Positively skewed data
                    self.df[column] = np.log(data + 1)
                elif skewness < -1:  # Negatively skewed data
                    self.df[column], _ = stats.boxcox(data[data > 0])
                elif 0 < skewness <= 1:  # Moderately positively skewed
                    self.df[column] = np.sqrt(data)
                elif -1 <= skewness < 0:  # Moderately negatively skewed
                    self.df[column], _ = stats.boxcox(data[data > 0])
        return self.check_normality()
        
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

     
    
        


def getVariability(df, column, threshold = 0.1):
    if (column in usefullColumns):
        return False
    column = df[column]
    
    if pd.api.types.is_numeric_dtype(column):
        variance = column.var()
        uniqueCount = len(column.unique())
        rangeValue = column.max() - column.min()
        print("variance", variance)
        print("unique", uniqueCount)
        print("range", rangeValue)
    else:
        print("Column is non-numeric, skipping variability check.")
        
    


def checkVariablity(df, column, threshold=0.1):
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
    else:
        print("Column is non-numeric, skipping variability check.")
        
    return False

def checkHighCardinality(column, threshold = 0.1):
    uniqueCount = column.nunique()
    totalCount = len(column)
    ratio = uniqueCount / totalCount
    return ratio > threshold

def checkCategoricalCardinality(column, thresholds=None):
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
        
    
    
def findingIrrelevantColumns(dataFrame):
    columnToRemove = []
    index = 0
    for column in dataFrame.columns:
        if (column in unique_identifiers):
            columnToRemove.append(column)
        elif (checkVariablity(dataFrame[column])):
            columnToRemove.append(column)
        index += 1
    return columnToRemove
    
        

def completeData(dataFrame):
    print("Null Values")
    print(nullCheck(dataFrame))
    print("Outliers")
    print(detectStandardOutliers(dataFrame))
    print("Irrelevant Columns")
    print(findingIrrelevantColumns(dataFrame))
