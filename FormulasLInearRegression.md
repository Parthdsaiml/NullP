

### **1. Linear Regression Equation**
The linear regression model assumes the relationship between the independent variable \(x\) and the dependent variable \(y\) is linear:

```
y = β₀ + β₁ * x + ε
```

Where:
- \( y \) is the dependent variable (the predicted value),
- \( x \) is the independent variable,
- \( β₀ \) is the intercept (the value of \(y\) when \(x = 0\)),
- \( β₁ \) is the slope (the change in \(y\) for a one-unit change in \(x\)),
- \( ε \) is the error term (the residual).

---

### **2. Formula for the Slope (β₁)**

To find the slope \( β₁ \) of the regression line, we use the formula:

```
β₁ = (n * Σ(xᵢ * yᵢ) - Σxᵢ * Σyᵢ) / (n * Σ(xᵢ²) - (Σxᵢ)²)
```

Where:
- \( n \) is the number of data points,
- \( Σxᵢ \) is the sum of all \(x\)-values,
- \( Σyᵢ \) is the sum of all \(y\)-values,
- \( Σ(xᵢ * yᵢ) \) is the sum of the product of \(x\) and \(y\) values,
- \( Σ(xᵢ²) \) is the sum of the squares of the \(x\)-values.

---

### **3. Formula for the Intercept (β₀)**

Once the slope \( β₁ \) is known, the intercept \( β₀ \) is calculated using the following formula:

```
β₀ = (Σyᵢ - β₁ * Σxᵢ) / n
```

Where:
- \( Σyᵢ \) is the sum of all \(y\)-values,
- \( β₁ \) is the slope (calculated from the previous formula),
- \( Σxᵢ \) is the sum of all \(x\)-values,
- \( n \) is the number of data points.

---

### **4. Predicted \( y \)-values (ŷ)**

The predicted \( y \)-values \( \hat{y} \) are calculated by plugging the \(x\)-values into the regression equation:

```
ŷᵢ = β₀ + β₁ * xᵢ
```

Where:
- \( ŷᵢ \) is the predicted value of \(y\) for a given \(xᵢ\),
- \( β₀ \) and \( β₁ \) are the intercept and slope, respectively,
- \( xᵢ \) is the independent variable value.

---

### **5. Total Sum of Squares (SST)**

SST measures the total variation in the observed data. It's the sum of squared differences between each observed \(y\)-value and the mean of the observed \(y\)-values (\( \bar{y} \)):

```
SST = Σ(yᵢ - ȳ)²
```

Where:
- \( yᵢ \) is each observed value,
- \( ȳ \) is the mean of the \(y\)-values,
- \( Σ \) indicates the sum over all data points.

---

### **6. Residual Sum of Squares (SSE)**

SSE measures the variation that is **unexplained** by the model (the sum of squared differences between observed \(y\)-values and predicted \(y\)-values):

```
SSE = Σ(yᵢ - ŷᵢ)²
```

Where:
- \( yᵢ \) is each observed value,
- \( ŷᵢ \) is the predicted value from the regression line.

---

### **7. \( R² \) (Coefficient of Determination)**

The coefficient of determination, \( R² \), tells us the proportion of variance in the dependent variable \(y\) that is explained by the independent variable \(x\):

```
R² = 1 - (SSE / SST)
```

Where:
- \( SSE \) is the residual sum of squares (unexplained variation),
- \( SST \) is the total sum of squares (total variation in the data).

Alternatively, it can also be written as:

```
R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

Where:
- \( yᵢ \) are the observed \(y\)-values,
- \( ŷᵢ \) are the predicted \(y\)-values,
- \( ȳ \) is the mean of the observed \(y\)-values.

---

### Summary of Formulas in ASCII:
1. **Linear Regression Model**:  
   `y = β₀ + β₁ * x + ε`

2. **Slope (β₁)**:  
   `β₁ = (n * Σ(xᵢ * yᵢ) - Σxᵢ * Σyᵢ) / (n * Σ(xᵢ²) - (Σxᵢ)²)`

3. **Intercept (β₀)**:  
   `β₀ = (Σyᵢ - β₁ * Σxᵢ) / n`

4. **Predicted y-values (ŷᵢ)**:  
   `ŷᵢ = β₀ + β₁ * xᵢ`

5. **Total Sum of Squares (SST)**:  
   `SST = Σ(yᵢ - ȳ)²`

6. **Residual Sum of Squares (SSE)**:  
   `SSE = Σ(yᵢ - ŷᵢ)²`

7. **\( R² \)**:  
   `R² = 1 - (SSE / SST)`

---

