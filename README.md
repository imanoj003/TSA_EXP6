# Ex.No: 6 HOLT WINTERS METHOD
## Developer name: Manoj M
## reg no : 212221240027
## Date: 13 / 10 / 24

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2.Install statsmodels if you don't already have it.
3. Prepare the data.
4. Apply the Holt-Winters method.
5. Forecast the next final grade (or smooth the existing grades).

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Create the student performance dataset
df = pd.read_csv('/content/student_performance.csv')

# Convert to DataFrame
df = pd.DataFrame(data)

# Treat StudentID as the time index for simplicity
df.set_index('StudentID', inplace=True)

# Plot the data
plt.plot(df.index, df['FinalGrade'], marker='o')
plt.title('Final Grades Over Student IDs')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.grid(True)
plt.show()

# Apply Holt-Winters method
# Using additive model (no seasonality, as the dataset is too small)
model = ExponentialSmoothing(df['FinalGrade'], trend='add', seasonal=None).fit()

# Predict the next 5 final grades (as an example)
predictions = model.forecast(steps=5)

# Print the fitted values and predictions
print("Fitted values:\n", model.fittedvalues)
print("\nPredictions for next 5 students:\n", predictions)

# Plot the results
plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')
plt.plot(df.index, model.fittedvalues, label='Fitted Values', linestyle='--')
plt.plot(range(len(df) + 1, len(df) + 6), predictions, label='Forecasted Values', marker='x')
plt.title('Holt-Winters Smoothing and Forecast')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:
# FINAL GRADES OVER STUDENT:
![Untitled](https://github.com/user-attachments/assets/62b9e31f-d00e-4f34-a7ec-09cafb276785)

# HOLT WINTERS SMOOTHING AND FORECAST:
![Untitled-1](https://github.com/user-attachments/assets/b4f461ee-cd56-4d8e-ae9b-39b95bc143f2)




### RESULT:
thus the program run successfully based on the Holt Winters Method model.
