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
    def __init__(self, df):
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying original data

    def isolation_forest_outliers(self, column, contamination=0.1):
        """
        Detect outliers using the Isolation Forest method.
        """
        data = column.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data)
        return np.where(outliers == -1)[0].tolist()

    def getIQRRange(self, column, dynamicValue):
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

    def countStandardOutliers(self, method='1', contamination=0.1):
        """
        Count the number of outliers in the dataframe using different methods: IQR, SD, Isolation Forest.
        """
        outliersResult = []
        skewedContainer = self.skewedDetection()
        for i, column in enumerate(self.df.columns):
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if method == '3':  # Isolation Forest
                    outliers = self.isolation_forest_outliers(self.df[column], contamination)
                    outliersResult.append([column, len(outliers)])
                elif skewedContainer[i] in [1, -1]:  # Skewed data
                    outliers = self.iqrOutliers(self.df[column])
                    outliersResult.append([column, len(outliers)])
                else:  # Non-skewed data
                    outliers = self.sdOutliers(self.df[column])
                    outliersResult.append([column, len(outliers)])
            else:
                outliersResult.append([column, None])  # For non-numeric columns
        return outliersResult

    def booleanOutliers(self, column, dynamicValue=-1):
        """
        Determine if there are any significant outliers based on the skewness and chosen method.
        """
        skewness = self.df[column].skew()
        if abs(skewness) > 0.5:
            outliers = self.iqrOutliers(self.df[column], dynamicValue)
        else:
            outliers = self.sdOutliers(self.df[column], dynamicValue)
        return len(outliers) > 0  # Returns True if there are outliers, False otherwise

    # Existing methods from the previous part would be here...

    def countOutliers(self):
        method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")

        # Initialize the list to hold constants if required
        dynamicConstant = []
        
        # Ask for constants only if method is IQR or SD
        if method in ['1', '2']:
            print("Enter the Constant for Each Column if you want to change the strictness of IQR or SD method or -1 for default")
            for column in self.df.columns:
                value = float(input(f"{column}: "))
                dynamicConstant.append(value)

        outliersResult = []  # This will hold the results as a list of lists
        skewedContainer = self.skewedDetection()
        i = 0
        
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if method == '1':  # IQR
                    outliers = self.iqrOutliers(self.df[column], dynamicConstant[i] if dynamicConstant else -1)
                    outliersResult.append([column, len(outliers)])
                elif method == '2':  # SD
                    outliers = self.sdOutliers(self.df[column], dynamicConstant[i] if dynamicConstant else -1)
                    outliersResult.append([column, len(outliers)])
                elif method == '3':  # Isolation Forest
                    contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
                    outliers = self.isolation_forest_outliers(self.df[column], contamination)
                    outliersResult.append([column, len(outliers)])
                
            else:
                outliersResult.append([column, None])  # Non-numeric columns can be handled as needed
            i += 1
        
        return outliersResult

    def detectStandardOutliers(self):
        method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")
        dynamicConstant = []
        if method == '3':
            contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
        
        outliersResult = []
        skewedContainer = self.skewedDetection()
        i = 0
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if method == '3':
                    outliers = self.isolation_forest_outliers(self.df[column], contamination)
                    outliersResult.append([column, len(outliers), outliers])
                elif skewedContainer[i] == 1 or skewedContainer[i] == -1:
                    outliers = self.iqrOutliers(self.df[column], -1)
                    outliersResult.append([column, len(outliers), outliers])
                else:
                    outliers = self.sdOutliers(self.df[column], -1)
                    outliersResult.append([column, len(outliers), outliers])
            else:
                outliersResult.append([column, None])  # Non-numeric columns
            i += 1
        
        return outliersResult

    def detectCategoricalOutliers(self, column_name, threshold_percent=1):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        column = self.df[column_name]
        category_counts = column.value_counts()
        threshold = len(column) * (threshold_percent / 100)
        outliers = category_counts[category_counts < threshold]
        return outliers.index.tolist(), outliers.values.tolist()  # Return categories and their counts

    def detectAllOutliers(self):
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
                    outliers = self.sd.outliers(self.df[column])
            elif (pd.api.types.is_object_dtype(self.df[column])):
                outliers = self.detectCategoricalOutliers(column)
                if (len(outliers) != 0):
                    allOutliers[column] = outliers
        return allOutliers

        
    def showOutliers(self, column, plot_type='boxplot'):
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

    def columnCountOutliers(self, column):
        if pd.api.types.is_numeric_dtype(column) == False:
            return None
        dynamicConstant = float(input("Enter the Constant for IQR or SD default is -1"))
        skew = column.skew()
        outLiersList = []
        if (skew < -0.5 or skew > 0.5):
            outliers = self.iqrOutliers(column, dynamicConstant)
            outLiersList.append(len(outliers)) 
        else:
            outliers = self.sdOutliers(column, dynamicConstant)
            outLiersList.append(len(outliers)) 
        
        return outLiersList

    def columnStandardGetOutliers(self, column):
        if pd.api.types.is_numeric_dtype(column) == False:
            return None
        
        skew = column.skew()
        if (skew < -0.5 or skew > 0.5):
            outliers = self.iqrOutliers(column, -1)
        else:
            outliers = self.sdOutliers(column, -1)
        # Create a DataFrame with the specified rows
        selected_df = self.df.iloc[outliers]
        
        return selected_df

    def isolation_forest_outliers(self, column, contamination=0.1):
        # Reshape the column to fit the model
        data = column.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data)
        
        # Return the indices of the outliers
        return np.where(outliers == -1)[0].tolist()

    def columnGetOutliers(self, column):
        if pd.api.types.is_numeric_dtype(self.df[column]) == False:
            return None
        dynamicConstant = float(input("Enter the Constant for IQR or SD default is -1"))
        
        skew = self.df[column].skew()
        if (skew < -0.5 or skew > 0.5):
            outliers = self.iqrOutliers(self.df[column], dynamicConstant)
        else:
            outliers = self.sdOutliers(self.df[column], dynamicConstant)
        
        # Create a DataFrame with the specified rows
        selected_df = self.df.iloc[outliers]
        
        return selected_df



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
