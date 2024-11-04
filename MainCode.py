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

def nullCheck(dataFrame):
    totalNullsCombineColumns = dataFrame.isnull().sum().sum()

    if (totalNullsCombineColumns != 0):
        print("Total Nulls : ", totalNullsCombineColumns)
        print("Null Count in Each Column")
        return dataFrame.isnull().sum()
        
    print("No Null Values found")
    # def deeperNull() to Be done
    return None

def skewCheck(column):
    skewness = column.skew()
    if (skewness < -0.5 or skewness > 0.5):
        return True
    else:
        return False
def replaceWithMode(df, column):
    modeValue = df[column].mode()[0]  # Get the most frequent value
    df[column] = df[column].fillna(modeValue)  # Avoid chained assignment
    print(f"Replaced with Mode, Column = [{column}]")
    return df

def replaceWithMean(df, column):
    meanValue = df[column].mean()
    df[column] = df[column].fillna(meanValue)
    print(f"Replaced with Mean, Column = [{column}]")
    return df
def replaceWithMedian(df, column):
    medianValue = df[column].median()
    df[column] = df[column].fillna(medianValue)  
    print(f"Replaced with Median, Column = [{column}]")
    return df


def knnMethodNullReplacment(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    return imputed_df


def automateNullUpdate(dataFrame, n_neighbors=5, percentNull=0.4, dropColumns=True):
    df_copy = dataFrame.copy()  # Avoid modifying the original DataFrame
    droppingColumns = []  # Initialize the list for dropped columns

    for column in df_copy.columns:
        nullCount = df_copy[column].isnull().sum()
        nullPercent = nullCount / len(df_copy[column])
        
        if nullPercent > percentNull:
            droppingColumns.append(column)
            continue
            
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            if booleanOutliers(df_copy, column):
                df_copy[column] = replaceWithMedian(df_copy, column)
        elif pd.api.types.is_string_dtype(df_copy[column]):
            # Replace missing string values with mode
            mode_value = df_copy[column].mode()[0]
            df_copy[column].fillna(mode_value, inplace=True)
    
    # Drop columns if specified
    if dropColumns:
        df_copy = df_copy.drop(columns=droppingColumns)
        print(f"Dropped columns with high null values: {droppingColumns}")

    # Apply KNN imputation for the entire DataFrame after handling outliers
    imputed_df = knn_method_null_replacement(df_copy, n_neighbors)
    return imputed_df

def replacingSimplerDataNullValues(df, percentNull=0.4, dropColumns=True):
    df_copy = df.copy()
    droppingColumns = []

    for column in df_copy.columns:
        nullCount = df_copy[column].isnull().sum()
        nullPercent = nullCount / len(df_copy[column])
        
        if nullPercent > percentNull:
            droppingColumns.append(column)
            continue
        
        if isCategorical(df_copy, column):
            df_copy = replaceWithMode(df_copy, column)
            continue
        
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            if skewCheck(df_copy[column]):
                df_copy = replaceWithMedian(df_copy, column)
            else:
                outliers = columnStandardCountOutliers(df_copy[column])
                if len(outliers) != 1 and outliers[0] == 0:
                    df_copy = replaceWithMedian(df_copy, column)
                else:
                    df_copy = replaceWithMean(df_copy, column)

    if dropColumns:
        if droppingColumns:  # Only drop if there are columns to drop
            df_copy = df_copy.drop(columns=droppingColumns)
            print(f"Dropped columns with high null values: {droppingColumns}")
        else:
            print("No columns were dropped.")

    return df_copy  # Return the modified DataFrame

def isCategorical(df, column_name):
    dtype = df[column_name].dtype
    return isinstance(dtype, pd.CategoricalDtype) or \
           (pd.api.types.is_object_dtype(dtype) and len(df[column_name].unique()) < 0.1 * len(df))

def visualizeMissingData(df, heatmap_color='viridis', save_fig=False, fig_prefix='missing_data'):
    if df.empty:
        print("DataFrame is empty.")
        return
    
    missing_data = df.isnull()
    
    # Create the heatmap for the entire dataset
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_data, cmap=heatmap_color, cbar=False, yticklabels=False, xticklabels=df.columns)
    plt.title('Missing Data Heatmap', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Observations', fontsize=14)
    
    if save_fig:
        plt.savefig(f"{fig_prefix}_heatmap.png", bbox_inches='tight')
    
    plt.show()

    # Count missing values per feature and plot as a bar chart
    missing_counts = df.isnull().sum()
    
    plt.figure(figsize=(10, 5))
    missing_counts.plot(kind='bar', color='skyblue')
    plt.title('Missing Values per Feature', fontsize=16)
    plt.ylabel('Number of Missing Values', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add percentages to the bar chart
    for index, value in enumerate(missing_counts):
        plt.text(index, value, f'{value} ({value/len(df)*100:.1f}%)', ha='center', va='bottom')

    if save_fig:
        plt.savefig(f"{fig_prefix}_missing_counts.png", bbox_inches='tight')
    
    plt.show()



def isolation_forest_outliers(column, contamination=0.1):
    # Reshape the column to fit the model
    data = column.values.reshape(-1, 1)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(data)
    
    # Return the indices of the outliers
    return np.where(outliers == -1)[0].tolist()


def getIQRRange(column, dynamicValue):
    sortedData = np.sort(column)
    length = len(sortedData)
    
    if len(sortedData) == 1:
        Q1 = sortedData[0]
        Q3 = sortedData[0]
    elif len(sortedData) == 2:
        Q1 = sortedData[0]
        Q3 = sortedData[1]
    else:
        Q1 = np.percentile(sortedData, 25)
        Q3 = np.percentile(sortedData, 75)
    IQR = Q3 - Q1
    if (dynamicValue != -1):
        
        lowerBound = Q1 - dynamicValue * IQR
        upperBound = Q3 + dynamicValue * IQR
        return [lowerBound, upperBound]
        
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    return [lowerBound, upperBound]
def iqrOutliers(column, valueDynamic):
    iqrRange = getIQRRange(column, valueDynamic)
    outlier_indices = []
    for idx, value in enumerate(column):
        if (value < iqrRange[0] or value > iqrRange[1]):
            outlier_indices.append(idx)  # Append the index of the outlier
    return outlier_indices
def sdRange(column, dynamicValue):
    meanValue = column.mean()
    stdValue = column.std()

    if (dynamicValue != -1):
        lowerRange = meanValue - (dynamicValue * stdValue)
        upperRange = meanValue + (dynamicValue * stdValue)
        return [lowerRange, upperRange]
        
    lowerRange = meanValue - (3 * stdValue)
    upperRange = meanValue + (3 * stdValue)
    return [lowerRange, upperRange]
def sdOutliers(column, valueDynamic):
    rangeSd = sdRange(column, valueDynamic)
    outLierIndices = []
    for idx, value in enumerate(column):
        if (value < rangeSd[0] or value > rangeSd[1]):
            outLierIndices.append(idx)
    return outLierIndices
def skewedDetection(dataFrame):
    
    
    skewedList = []
    skew = 0
    for column in dataFrame.columns:
        if pd.api.types.is_numeric_dtype(dataFrame[column]):
            if (dataFrame[column].nunique() < 5):
                skewedList.append(None)
                continue
            skew = dataFrame[column].skew()
            if (skew > 0.5):
                skewedList.append(1)
            elif (skew < -0.5):
                skewedList.append(-1)
            else:
                skewedList.append(0)
        else:
            skewedList.append(None)
    return skewedList

        
            

def countStandardOutliers(dataFrame):
    method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")
    dynamicConstant = []
    if method == '3':
        contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
    
    outliersResult = []
    skewedContainer = skewedDetection(dataFrame)
    i = 0
    for column in dataFrame.columns:
        if pd.api.types.is_numeric_dtype(dataFrame[column]):
            if method == '3':
                outliers = isolation_forest_outliers(dataFrame[column], contamination)
                outliersResult.append([column, len(outliers)])
            elif skewedContainer[i] == 1 or skewedContainer[i] == -1:
                outliers = iqrOutliers(dataFrame[column], -1)
                outliersResult.append([column, len(outliers)])
            else:
                outliers = sdOutliers(dataFrame[column], -1)
                outliersResult.append([column, len(outliers)])
        else:
            outliersResult.append([column, None])  # Non-numeric columns
        i += 1
        
    return outliersResult

def booleanOutliers(dataFrame, column, dynamicValue = -1):
    skewness = dataFrame[column].skew();
    if (skewness < -0.5 or skewness > 0.5):
        outliers = iqrOutliers(dataFrame[column], dynamicValue)
    else:
        outliers = sdOutliers(dataFrame[column], dynamicValue)
    if (len(outliers) == 1 and outliers[0] == 0):
        return False
    else:
        return True
    
def countOutliers(dataFrame):
    method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")

    # Initialize the list to hold constants if required
    dynamicConstant = []
    
    # Ask for constants only if method is IQR or SD
    if method in ['1', '2']:
        print("Enter the Constant for Each Column if you want to change the strictness of IQR or SD method or -1 for default")
        for column in dataFrame.columns:
            value = float(input(f"{column}: "))
            dynamicConstant.append(value)

    outliersResult = []  # This will hold the results as a list of lists
    skewedContainer = skewedDetection(dataFrame)
    i = 0
    
    for column in dataFrame.columns:
        if pd.api.types.is_numeric_dtype(dataFrame[column]):
            if method == '1':  # IQR
                outliers = iqrOutliers(dataFrame[column], dynamicConstant[i] if dynamicConstant else -1)
                outliersResult.append([column, len(outliers)])
            elif method == '2':  # SD
                outliers = sdOutliers(dataFrame[column], dynamicConstant[i] if dynamicConstant else -1)
                outliersResult.append([column, len(outliers)])
            elif method == '3':  # Isolation Forest
                contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
                outliers = isolation_forest_outliers(dataFrame[column], contamination)
                outliersResult.append([column, len(outliers)])
                
        else:
            outliersResult.append([column, None])  # Non-numeric columns can be handled as needed
        i += 1
        
    return outliersResult

def detectStandardOutliers(dataFrame):
    method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")
    dynamicConstant = []
    if method == '3':
        contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
    
    outliersResult = []
    skewedContainer = skewedDetection(dataFrame)
    i = 0
    for column in dataFrame.columns:
        if pd.api.types.is_numeric_dtype(dataFrame[column]):
            if method == '3':
                outliers = isolation_forest_outliers(dataFrame[column], contamination)
                outliersResult.append([column, len(outliers), outliers])
            elif skewedContainer[i] == 1 or skewedContainer[i] == -1:
                outliers = iqrOutliers(dataFrame[column], -1)
                outliersResult.append([column, len(outliers), outliers])
            else:
                outliers = sdOutliers(dataFrame[column], -1)
                outliersResult.append([column, len(outliers), outliers])
        else:
            outliersResult.append([column, None])  # Non-numeric columns
        i += 1
        
    return outliersResult


def detectOutliers(dataFrame):
    method = input("Choose outlier detection method (1: IQR, 2: SD, 3: Isolation Forest): ")

    # Initialize the list to hold constants if required
    dynamicConstant = []
    
    # Ask for constants only if method is IQR or SD
    if method in ['1', '2']:
        print("Enter the Constant for Each Column if you want to change the strictness of IQR or SD method or -1 for default")
        for column in dataFrame.columns:
            value = float(input(f"{column}: "))
            dynamicConstant.append(value)

    outliersResult = []  # This will hold the results as a list of lists
    skewedContainer = skewedDetection(dataFrame)
    i = 0
    
    for column in dataFrame.columns:
        if pd.api.types.is_numeric_dtype(dataFrame[column]):
            if method == '1':  # IQR
                outliers = iqrOutliers(dataFrame[column], dynamicConstant[i] if dynamicConstant else -1)
                outliersResult.append([column, len(outliers), outliers])
            elif method == '2':  # SD
                outliers = sdOutliers(dataFrame[column], dynamicConstant[i] if dynamicConstant else -1)
                outliersResult.append([column, len(outliers), outliers])
            elif method == '3':  # Isolation Forest
                contamination = float(input("Enter contamination level for Isolation Forest (default 0.1): "))
                outliers = isolation_forest_outliers(dataFrame[column], contamination)
                outliersResult.append([column, len(outliers), outliers])
                
        else:
            outliersResult.append([column, None])  # Non-numeric columns can be handled as needed
        i += 1
        
    return outliersResult

def showOutliers(dataFrame, column, plot_type='boxplot'):
    # Check if the column exists in the DataFrame
    if column not in dataFrame.columns:
        print(f"Column '{column}' does not exist in the DataFrame.")
        return
    
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'boxplot':
        # Create a box plot
        sns.boxplot(y=dataFrame[column])
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)

    elif plot_type == 'scatter':
        # Create a scatter plot
        plt.scatter(dataFrame.index, dataFrame[column])
        plt.title(f'Scatter Plot of {column}')
        plt.ylabel(column)
        plt.xlabel('Index')

    elif plot_type == 'histogram':
        # Create a histogram
        sns.histplot(dataFrame[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    else:
        print(f"Plot type '{plot_type}' is not supported.")
        return
    
    plt.show()

    
    
def columnStandardCountOutliers(column):
    if pd.api.types.is_numeric_dtype(column) == False:
        return None
    skew = column.skew()
    outLiersList = []
    if (skew < -0.5 or skew > 0.5):
        outliers = iqrOutliers(column, -1)
        outLiersList.append(len(outliers)) 
    else:
        outliers = sdOutliers(column, -1)
        outLiersList.append(len(outliers)) 
        
    return outLiersList
def columnCountOutliers(column):
    if pd.api.types.is_numeric_dtype(column) == False:
        return None
    dynamicConstant = float(input("Enter the Constant for IQR or SD default is -1"))
    skew = column.skew()
    outLiersList = []
    if (skew < -0.5 or skew > 0.5):
        outliers = iqrOutliers(column, dynamicConstant)
        outLiersList.append(len(outliers)) 
    else:
        outliers = sdOutliers(column, dynamicConstant)
        outLiersList.append(len(outliers)) 
        
    return outLiersList

def columnStandardGetOutliers(dataFrame, column):
    if pd.api.types.is_numeric_dtype(column) == False:
        return None
    
    skew = column.skew()
    if (skew < -0.5 or skew > 0.5):
        outliers = iqrOutliers(column, -1)
    else:
        outliers = sdOutliers(column, -1)
    # Create a DataFrame with the specified rows
    selected_df = dataFrame.iloc[outliers]
    

    return selected_df
def columnGetOutliers(dataFrame, column):
    if pd.api.types.is_numeric_dtype(column) == False:
        return None
    dynamicConstant = float(input("Enter the Constant for IQR or SD default is -1"))
    
    skew = column.skew()
    if (skew < -0.5 or skew > 0.5):
        outliers = iqrOutliers(column, dynamicConstant)
    else:
        outliers = sdOutliers(column, dynamicConstant)
    # Create a DataFrame with the specified rows
    selected_df = dataFrame.iloc[outliers]
    

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
