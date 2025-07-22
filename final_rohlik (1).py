#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import os

# zip file paths and extracted path
zip_path = r'C:\Users\91720\OneDrive\Desktop\exafluence\rohlik-sales-forecasting-challenge-v2.zip'
extract_path = r'C:\Users\91720\OneDrive\Desktop\exafluence'


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


file_list = os.listdir(extract_path)
print("Extracted files:", file_list)


# In[2]:


import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import warnings
warnings.filterwarnings('ignore')


# In[3]:


sales_path = os.path.join(extract_path, 'sales_train.csv')
train_df = pd.read_csv(sales_path)
train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
test_df = pd.read_csv(os.path.join(extract_path, 'sales_test.csv'))
weights_df = pd.read_csv(os.path.join(extract_path, 'test_weights.csv'))
df5 = pd.read_csv(os.path.join(extract_path, 'solution.csv'))

inventory_df = pd.read_csv(os.path.join(extract_path, 'inventory.csv'))
calendar_df = pd.read_csv(os.path.join(extract_path, 'calendar.csv'))


# In[4]:


test_columns = list(test_df.columns)
keep_columns =  list(train_df.columns)
print(test_columns)
keep_columns 


# In[44]:


# EDA  for calendar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

calendar_df = pd.read_csv(os.path.join(extract_path, 'calendar.csv'))
calendar_df.columns = calendar_df.columns.str.lower().str.strip()

# Convert date to datetime
calendar_df['date'] = pd.to_datetime(calendar_df['date'], errors='coerce')

# data Summary 
print("📌 Shape:", calendar_df.shape)
print("📌 Columns:", calendar_df.columns.tolist())
print("\n📌 Info:")
print(calendar_df.info())
print("\n📌 Missing Values:")
print(calendar_df.isnull().sum())
print("\n📌 Description:")
print(calendar_df.describe(include='all'))
print("\n📌 Sample Rows:")
print(calendar_df.head())

# Top Holidays
top_holidays = calendar_df['holiday_name'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_holidays.index, x=top_holidays.values, palette='Set2')
plt.title("Top 10 Most Frequent Holidays")
plt.xlabel("Count")
plt.ylabel("Holiday Name")
plt.tight_layout()
plt.show()

# Holidays per Warehouse
plt.figure(figsize=(10, 6))
sns.countplot(data=calendar_df[calendar_df['holiday'] == 1], x='warehouse', order=calendar_df['warehouse'].value_counts().index, palette='Set3')
plt.title("Holidays per Warehouse")
plt.xlabel("Warehouse")
plt.ylabel("Number of Holidays")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly School vs Winter Holidays Trend
monthly_holidays = calendar_df.groupby(calendar_df['date'].dt.to_period('M'))[
    ['school_holidays', 'winter_school_holidays']
].sum()

monthly_holidays.plot(kind='bar', figsize=(14, 5), color=['teal', 'orange'])
plt.title("Monthly School and Winter Holidays Trend")
plt.xlabel("Month")
plt.ylabel("Number of Holidays")
plt.tight_layout()
plt.show()

# Shops Closed vs Holidays Count
closed_vs_holiday = calendar_df[['shops_closed', 'holiday']].sum()

closed_vs_holiday.plot(kind='bar', color=['red', 'green'])
plt.title("Comparison: Shops Closed vs Holidays")
plt.ylabel("Number of Days")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Warehouse Entry Count
warehouse_counts = calendar_df['warehouse'].value_counts()

warehouse_counts.plot(kind='bar', color='skyblue')
plt.title("Number of Calendar Entries per Warehouse")
plt.xlabel("Warehouse")
plt.ylabel("Entry Count")
plt.tight_layout()
plt.show()


# In[45]:


# EDA for inventory.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data
inventory_df = pd.read_csv(os.path.join(extract_path, 'inventory.csv'))
inventory_df.columns = inventory_df.columns.str.lower().str.strip()

# Product Frequency by Name
top_products = inventory_df['name'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_products.index, x=top_products.values, palette='Blues_d')
plt.title("Top 10 Most Frequent Product Names")
plt.xlabel("Count")
plt.ylabel("Product Name")
plt.tight_layout()
plt.show()

# Distribution by Level 1 Category
plt.figure(figsize=(10, 4))
sns.countplot(data=inventory_df, y='l1_category_name_en', order=inventory_df['l1_category_name_en'].value_counts().index, palette='Set2')
plt.title("Product Distribution by Level 1 Category")
plt.xlabel("Count")
plt.ylabel("L1 Category")
plt.tight_layout()
plt.show()

# Distribution by Level 2 Category (only the top 10 values)
top_l2 = inventory_df['l2_category_name_en'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(y=top_l2.index, x=top_l2.values, palette='viridis')
plt.title("Top 10 L2 Categories by Product Count")
plt.xlabel("Count")
plt.ylabel("L2 Category")
plt.tight_layout()
plt.show()

# Distribution by Warehouse
warehouse_counts = inventory_df['warehouse'].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=warehouse_counts.index, y=warehouse_counts.values, palette='coolwarm')
plt.title("Inventory Count by Warehouse")
plt.xlabel("Warehouse")
plt.ylabel("Count of Products")
plt.tight_layout()
plt.show()


# In[46]:


# EDA for sales_train.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
sales_df = pd.read_csv(os.path.join(extract_path, 'sales_train.csv'))
sales_df.columns = sales_df.columns.str.lower().str.strip()

# Summary
print("📌 Shape:", sales_df.shape)
print("📌 Columns:", sales_df.columns.tolist())
print("\n📌 Info:")
print(sales_df.info())
print("\n📌 Missing Values:")
print(sales_df.isnull().sum())
print("\n📌 Description:")
print(sales_df.describe(include='all'))
print("\n📌 Sample Rows:")
print(sales_df.head())

#  Sales Distribution
if 'sales' in sales_df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(sales_df['sales'], bins=100, kde=True)
    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'sales' column not found.")

# Top 10 Products by Total Sales count
if 'unique_id' in sales_df.columns:
    top_products = sales_df.groupby('unique_id')['sales'].sum().sort_values(ascending=False).head(10)
    top_products.plot(kind='bar', color='teal')
    plt.title("Top 10 Products by Total Sales")
    plt.ylabel("Total Sales")
    plt.xlabel("Product ID")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'product_id' column not found.")

#  Sales Over Time 
if 'date' in sales_df.columns:
    sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')
    daily_sales = sales_df.groupby('date')['sales'].sum()

    plt.figure(figsize=(14, 5))
    daily_sales.plot()
    plt.title("Total Sales Over Time")
    plt.ylabel("Sales")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'date' column not found.")

#  Sales by Warehouse
if 'warehouse' in sales_df.columns:
    warehouse_sales = sales_df.groupby('warehouse')['sales'].sum().sort_values(ascending=False)
    warehouse_sales.plot(kind='bar', color='coral')
    plt.title("Total Sales by Warehouse")
    plt.xlabel("Warehouse")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

# Top Selling Days
if 'date' in sales_df.columns:
    top_days = sales_df.groupby('date')['sales'].sum().sort_values(ascending=False).head(10)
    top_days.plot(kind='bar', color='purple')
    plt.title("Top 10 Highest Sales Days")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()


# In[5]:


from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# --- Create dictionary of scalers for train_df ---
scalers = {}
for unique_id in tqdm(train_df["unique_id"].unique()):
    scaler = StandardScaler()
    sales = train_df.loc[train_df["unique_id"] == unique_id, "sales"].values.reshape(-1, 1)
    scaler.fit(sales)
    scalers[unique_id] = scaler
    train_df.loc[train_df["unique_id"] == unique_id, "sales"] = scaler.transform(sales).flatten()


# In[6]:


def inverse_norm(df_, indexes, y_pred, scalers):
    df_ = df_.copy()  # Avoid modifying original DataFrame
    
    # Temporarily assign predictions to a new column
    df_.loc[indexes, "prediction_norm"] = y_pred
    w
    # Inverse transform normalized predictions per unique_id
    df_.loc[indexes, "y_pred"] = df_.groupby("unique_id")["prediction_norm"].transform(
        lambda x: scalers[x.name].inverse_transform(x.values.reshape(-1, 1)).flatten()
        if x.name in scalers else x.values  # If no scaler, return unchanged
    )
    
    return df_.loc[indexes, "y_pred"].values


# In[7]:


#eda
# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')

# Filter rows where sales >= 0
train_df_copy = train_df[train_df['sales'] >= 0]

# Aggregate sales by day
daily_sales = train_df_copy.groupby('date')['sales'].sum().reset_index()

# Set Seaborn style for better visuals
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Plot line chart
plt.figure(figsize=(15, 5))
sns.lineplot(data=daily_sales, x='date', y='sales', label='Total Daily Sales')

plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set(style="whitegrid")

# Create a figure with 2 subplots (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram of sales
sns.histplot(data=train_df, x="sales", bins=30, kde=True, ax=ax1)
ax1.set_title("Histogram of Sales")
ax1.set_xlabel("Sales")
ax1.set_ylabel("Frequency")

# Plot boxplot of sales
sns.boxplot(data=train_df, y="sales", ax=ax2)
ax2.set_title("Boxplot of Sales")
ax2.set_ylabel("Sales")

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

# Convert 'date' to datetime format (if not already done)
calendar_df['date'] = pd.to_datetime(calendar_df['date'], errors='coerce')

# Add a 'month_n' column (numeric month)
calendar_df['month_n'] = calendar_df['date'].dt.month

# Aggregate data by warehouse and month
grouped_df = calendar_df.groupby(['warehouse', 'month_n'])[
    ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']
].sum().reset_index()

# Get list of unique warehouses
warehouses = grouped_df['warehouse'].unique()

# Set up subplot grid
rows = (len(warehouses) + 2) // 3  # 3 columns per row
fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(18, rows * 4))
axes = axes.flatten()

# Colors for each category
colors = ['blue', 'orange', 'green', 'pink']

# Plot for each warehouse
for i, warehouse in enumerate(warehouses):
    data = grouped_df[grouped_df['warehouse'] == warehouse].set_index('month_n')
    data = data[['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']]

    data.plot(kind='bar', stacked=True, ax=axes[i], color=colors)
    axes[i].set_title(f'{warehouse}', fontsize=16)
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Number of Days')
    axes[i].legend(loc='upper right', fontsize=10)
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].set_xticks(range(0, 12))

# Remove any empty subplots
for j in range(len(warehouses), len(axes)):
    fig.delaxes(axes[j])

# Set main title
fig.suptitle("Monthly Distribution of Holidays and School Closures by Warehouse", fontsize=18)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[10]:


#format date
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# In[11]:


# Filter data for each warehouse starting from January 1, 2016
Frankfurt_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Frankfurt_1"')
Prague_2 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_2"')
Brno_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Brno_1"')
Munich_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Munich_1"')
Prague_3 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_3"')
Prague_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_1"')
Budapest_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Budapest_1"')
def process_calendar(df):
    """
    - days_to_holiday
    - days_to_shops_closed
    - day_after_closing
    - long_weekend
    - weekday
    - ...
    """
    df = df.sort_values('date').reset_index(drop=True)


    # 4. long_weekend 
  
    df['long_weekend'] = (
        (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)

    # 5. weekday
    df['weekday'] = df['date'].dt.weekday 

    # 6. week of month 
    df['week_of_month'] = df['date'].apply(lambda x: (x.day - 1) // 7 + 1)

    # 9. quarter
    df['quarter'] = df['date'].dt.quarter

    # 10. is weekend
    df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

    return df


dfs = ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']

processed_dfs = [process_calendar(globals()[df]) for df in dfs]

calendar_extended = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)
print(calendar_extended.isna().sum())


# In[12]:


#merged the dataset

train_calendar = train_df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
train_inventory = train_calendar.merge(inventory_df, on=['unique_id', 'warehouse'], how='left')
train_df = train_inventory.merge(weights_df, on=['unique_id'], how='left')

test_calendar = test_df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
test_df = test_calendar.merge(inventory_df, on=['unique_id', 'warehouse'], how='left')


# In[13]:



train_df['sales'] = train_df['sales'].fillna(0)
train_df['total_orders'] = train_df['total_orders'].fillna(0)
train_df['sell_price_main'] = train_df['sell_price_main'].interpolate()


# In[14]:


print("Number of NaN values in 'holiday_name' grouped by each 'holiday':")
print(train_df.groupby('holiday')['holiday_name'].apply(lambda x: x.isna().sum()))


# In[15]:


# Optional: Drop old columns if you want to use calendar_df's holiday info
train_df.drop(columns=['holiday', 'holiday_name'], errors='ignore', inplace=True)

# Then merge
train_df = train_df.merge(
    calendar_df[['date', 'warehouse', 'holiday', 'holiday_name']],
    on=['date', 'warehouse'],
    how='left'
)


# In[ ]:





# In[16]:


try:
    # Check if 'sales_df' is defined
    if 'train_df' not in globals():
        raise NameError("train_df is not defined. Please ensure it has been created earlier in the code.")
    
    # Check if required columns exist
    required_columns = ['date', 'warehouse', 'holiday', 'holiday_name']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"train_df is missing the following columns: {missing_columns}")

    # Filter for holidays that have missing holiday names (holiday == 1 and holiday_name is NaN)
    missing_holidays = train_df[(train_df['holiday'] == 1) & (train_df['holiday_name'].isna())][['date', 'warehouse']]

    # If no such data exists
    if missing_holidays.empty:
        print("No holidays found with missing names (holiday == 1 and holiday_name is NaN).")
    else:
        # Group by warehouse and get sorted list of unique missing holiday dates
        missing_by_warehouse = missing_holidays.groupby('warehouse').agg({
            'date': lambda x: sorted(x.dt.strftime('%Y-%m-%d').unique().tolist())
        }).reset_index()

        # Rename columns for clarity
        missing_by_warehouse.columns = ['warehouse', 'missing_holiday_dates']

        # Print results
        print("Holidays with missing names by warehouse (sorted by date):")
        for _, row in missing_by_warehouse.iterrows():
            warehouse = row['warehouse']
            dates = row['missing_holiday_dates']
            print(f"\nWarehouse: {warehouse}")
            print(f"Number of unnamed holidays: {len(dates)}")
            print("Dates with missing holiday name:", ", ".join(dates) if dates else "None")

except NameError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


# In[17]:


from datetime import datetime

# 🔹 Define known holiday dates for each warehouse
brno_holiday = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother's Day"),
]

prague_1_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]

prague_2_holidays = prague_1_holidays
prague_3_holidays = prague_1_holidays

budapest_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]

frank_holidays = [
    (['04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], "Mother's Day"),
]

munich_holidays = [
    (['04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]

# 🔹 Function to fill in missing holiday information
def fill_missing_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

# 🔹 Validate that dataframes exist and contain necessary columns
try:
    if 'train_df' not in globals() or 'test_df' not in globals():
        raise NameError("train_df or test_df is not defined. Please make sure both are created beforehand.")

    required_columns = ['date', 'warehouse']
    for df_name, df in [('train_df', train_df), ('test_df', test_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{df_name} is missing the following columns: {missing_columns}")

        # Create 'holiday' and 'holiday_name' columns if missing
        if 'holiday' not in df.columns:
            df['holiday'] = 0
        if 'holiday_name' not in df.columns:
            df['holiday_name'] = None

    # 🔹 Print how many holiday_name values are missing (before processing)
    print("Number of NaN values in holiday_name (train_df) before processing:", train_df['holiday_name'].isna().sum())
    print("\nNaN count in holiday_name grouped by holiday (train_df):")
    print(train_df.groupby('holiday')['holiday_name'].apply(lambda x: x.isna().sum()))

    # 🔹 Fill missing holidays in train_df
    train_df = fill_missing_holidays(train_df, ['Prague_1', 'Prague_2', 'Prague_3'], prague_1_holidays)
    train_df = fill_missing_holidays(train_df, ['Brno_1'], brno_holiday)
    train_df = fill_missing_holidays(train_df, ['Munich_1'], munich_holidays)
    train_df = fill_missing_holidays(train_df, ['Frankfurt_1'], frank_holidays)
    train_df = fill_missing_holidays(train_df, ['Budapest_1'], budapest_holidays)

    train_df['holiday_name'] = train_df['holiday_name'].fillna("No Holiday")

    print("\nNumber of NaN in holiday_name (train_df) after processing:", train_df['holiday_name'].isna().sum())
    print("\nValue distribution in holiday_name (train_df):")
    print(train_df['holiday_name'].value_counts())
    print("\nValues of holiday_name where holiday == 1 (train_df):")
    print(train_df[train_df['holiday'] == 1]['holiday_name'].value_counts())

    # 🔹 Repeat the same for test_df
    print("\nNumber of NaN values in holiday_name (test_df) before processing:", test_df['holiday_name'].isna().sum())
    test_df = fill_missing_holidays(test_df, ['Prague_1', 'Prague_2', 'Prague_3'], prague_1_holidays)
    test_df = fill_missing_holidays(test_df, ['Brno_1'], brno_holiday)
    test_df = fill_missing_holidays(test_df, ['Munich_1'], munich_holidays)
    test_df = fill_missing_holidays(test_df, ['Frankfurt_1'], frank_holidays)
    test_df = fill_missing_holidays(test_df, ['Budapest_1'], budapest_holidays)

    test_df['holiday_name'] = test_df['holiday_name'].fillna("No Holiday")

    print("\nNumber of NaN in holiday_name (test_df) after processing:", test_df['holiday_name'].isna().sum())
    print("\nValue distribution in holiday_name (test_df):")
    print(test_df['holiday_name'].value_counts())

except NameError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


# In[18]:


from scipy.stats import zscore

log_sales = np.log1p(train_df['sales'])
z_scores = zscore(log_sales)
outliers = np.abs(z_scores) > 3  # Common threshold

print(f"Outlier count (z-score > 3): {outliers.sum()}")


# In[ ]:





# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate IQR bounds
def get_iqr_bounds(log_series):
    Q1 = log_series.quantile(0.25)
    Q3 = log_series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

# Function to plot all types
def plot_distributions(before_log, after, feature_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Boxplot
    sns.boxplot(x=before_log, ax=axes[0, 0])
    axes[0, 0].set_title(f"{feature_name} (log1p) - Before Capping")

    sns.boxplot(x=np.log1p(after), ax=axes[0, 1])
    axes[0, 1].set_title(f"{feature_name} (log1p) - After Capping")
    
    # Histogram + KDE
    sns.histplot(before_log, kde=True, ax=axes[1, 0], bins=30, color='skyblue')
    axes[1, 0].set_title(f"{feature_name} Distribution - Before Capping")

    sns.histplot(np.log1p(after), kde=True, ax=axes[1, 1], bins=30, color='lightgreen')
    axes[1, 1].set_title(f"{feature_name} Distribution - After Capping")

    plt.tight_layout()
    plt.show()

# -----------------------------
# Handle 'sales' with capping
# -----------------------------
log_sales = np.log1p(train_df['sales'])
lower_sales, upper_sales = get_iqr_bounds(log_sales)

# Cap
train_df['sales'] = np.expm1(np.clip(log_sales, lower_sales, upper_sales))

# Plot before/after
plot_distributions(log_sales, train_df['sales'], 'Sales')

# -----------------------------
# Handle 'total_orders' with capping
# -----------------------------
log_orders = np.log1p(train_df['total_orders'])
lower_orders, upper_orders = get_iqr_bounds(log_orders)

# Cap
train_df['total_orders'] = np.expm1(np.clip(log_orders, lower_orders, upper_orders))

# Plot before/after
plot_distributions(log_orders, train_df['total_orders'], 'Total Orders')


# In[20]:


# List of discount columns
discount_columns = [f"type_{i}_discount" for i in range(7)]

# Filter rows where any discount column has a negative value
negative_discounts = train_df[train_df[discount_columns].lt(0).any(axis=1)]

# Display rows with negative values
if not negative_discounts.empty:
    print("Rows with negative values in discount columns:")
    print(negative_discounts[["unique_id", "date"] + discount_columns])
    
    # Count how many negative values exist in each discount column
    print("\nNumber of negative values in each discount column:")
    for col in discount_columns:
        num_negative = len(train_df[train_df[col] < 0])
        print(f"{col}: {num_negative}")
else:
    print("No negative values found in any discount column.")


# In[21]:


train_df.loc[train_df['type_0_discount'] < 0, 'type_0_discount'] = 0
train_df.loc[train_df['type_4_discount'] < 0, 'type_4_discount'] = 0
train_df.loc[train_df['type_6_discount'] < 0, 'type_6_discount'] = 0


# In[22]:


train_df.head()


# In[23]:


test_df['sales'] = 0.0
test_df.info()


# In[24]:


print (train_df['L1_category_name_en'].unique())


# In[25]:


# 1. Calculate average category sales for train_df and map to train_df
category_sales_train = train_df.groupby('L1_category_name_en')['sales'].mean().reset_index()
category_sales_train.rename(columns={'sales': 'category_sales_avg'}, inplace=True)
train_df['category_sales_avg'] = train_df['L1_category_name_en'].map(
    category_sales_train.set_index('L1_category_name_en')['category_sales_avg']
)

# 2. Map the same category_sales from train to test_df (based on same logic)
test_df['category_sales_avg'] = test_df['L1_category_name_en'].map(
    category_sales_train.set_index('L1_category_name_en')['category_sales_avg']
)

# 3. Calculate average total orders per category for train_df and map to train_df
cat_order_stats_train = train_df.groupby('L1_category_name_en')['total_orders'].mean().reset_index()
cat_order_stats_train.rename(columns={'total_orders': 'category_orders_avg'}, inplace=True)
train_df['category_orders_avg'] = train_df['L1_category_name_en'].map(
    cat_order_stats_train.set_index('L1_category_name_en')['category_orders_avg']
)

# 4. Map the same category_orders_avg from train to test_df
test_df['category_orders_avg'] = test_df['L1_category_name_en'].map(
    cat_order_stats_train.set_index('L1_category_name_en')['category_orders_avg']
)

# Convert 'date' to datetime and extract features
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear

# Handle NaN in test_df (if needed)
test_df['category_sales_avg'].fillna(0.0, inplace=True)  # Because sales in test_df are 0
test_df['category_orders_avg'].fillna(test_df['total_orders'].mean(), inplace=True)


# In[26]:


# Calculate the maximum discount applied in Sales_Train

discount_columns = [f'type_{i}_discount' for i in range(7)]
train_df['max_discount'] = train_df[discount_columns].max(axis=1)

# Calculate the maximum discount applied in Sales_Test
discount_columns_test = [f'type_{i}_discount' for i in range(7)]
test_df['max_discount'] = test_df[discount_columns_test].max(axis=1)


# In[27]:


train_df.info()
test_df.info()


# In[28]:


from sklearn.preprocessing import LabelEncoder

# ✅ Check and convert 'date' column to datetime
if 'date' not in train_df.columns:
    raise ValueError("Column 'date' does not exist in train_df")
if 'date' not in test_df.columns:
    raise ValueError("Column 'date' does not exist in test_df")

train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')
test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')

# ✅ Validate that the conversion to datetime was successful
if train_df['date'].isna().all():
    raise ValueError("Column 'date' in train_df does not contain valid datetime values")
if test_df['date'].isna().all():
    raise ValueError("Column 'date' in test_df does not contain valid datetime values")

# ✅ Extract time features from 'date' for train_df
train_df['dayofyear'] = train_df['date'].dt.dayofyear
train_df["day"] = train_df["date"].dt.day
train_df["month"] = train_df["date"].dt.month
train_df["year"] = train_df["date"].dt.year

# ✅ Create cyclical features using sine and cosine to capture temporal patterns
train_df['year_sin'] = np.sin(2 * np.pi * train_df['year'] / train_df['year'].max())
train_df['year_cos'] = np.cos(2 * np.pi * train_df['year'] / train_df['year'].max())
train_df['month_sin'] = np.sin(2 * np.pi * train_df['month'] / 12)
train_df['month_cos'] = np.cos(2 * np.pi * train_df['month'] / 12)
train_df['day_sin'] = np.sin(2 * np.pi * train_df['day'] / 31)
train_df['day_cos'] = np.cos(2 * np.pi * train_df['day'] / 31)

# ✅ 1. Calculate days to next holiday
train_df['next_holiday_date'] = train_df.loc[train_df['holiday'] == 1, 'date'].shift(-1)
train_df['next_holiday_date'] = train_df['next_holiday_date'].bfill()
train_df['days_to_holiday'] = (train_df['next_holiday_date'] - train_df['date']).dt.days
train_df.drop(columns=['next_holiday_date'], inplace=True)

# ✅ 2. Calculate days to next shop closure
train_df['next_shops_closed_date'] = train_df.loc[train_df['shops_closed'] == 1, 'date'].shift(-1)
train_df['next_shops_closed_date'] = train_df['next_shops_closed_date'].bfill()
train_df['days_to_shops_closed'] = (train_df['next_shops_closed_date'] - train_df['date']).dt.days
train_df.drop(columns=['next_shops_closed_date'], inplace=True)

# ✅ 3. Mark the day after the store was closed
train_df['day_after_closing'] = (
    (train_df['shops_closed'] == 0) & (train_df['shops_closed'].shift(1) == 1)
).astype(int)

# ✅ 4. Mark long weekends (consecutive shop closures)
train_df['long_weekend'] = (
    (train_df['shops_closed'] == 1) & (train_df['shops_closed'].shift(1) == 1)
).astype(int)

# ✅ Repeat the same steps for test_df
test_df['dayofyear'] = test_df['date'].dt.dayofyear
test_df["day"] = test_df["date"].dt.day
test_df["month"] = test_df["date"].dt.month
test_df["year"] = test_df["date"].dt.year
test_df['year_sin'] = np.sin(2 * np.pi * test_df['year'] / test_df['year'].max())
test_df['year_cos'] = np.cos(2 * np.pi * test_df['year'] / test_df['year'].max())
test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] / 12)
test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] / 12)
test_df['day_sin'] = np.sin(2 * np.pi * test_df['day'] / 31)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day'] / 31)

# ✅ 1. Calculate days to next holiday
test_df['next_holiday_date'] = test_df.loc[test_df['holiday'] == 1, 'date'].shift(-1)
test_df['next_holiday_date'] = test_df['next_holiday_date'].bfill()
test_df['days_to_holiday'] = (test_df['next_holiday_date'] - test_df['date']).dt.days
test_df.drop(columns=['next_holiday_date'], inplace=True)

# ✅ 2. Calculate days to next shop closure
test_df['next_shops_closed_date'] = test_df.loc[test_df['shops_closed'] == 1, 'date'].shift(-1)
test_df['next_shops_closed_date'] = test_df['next_shops_closed_date'].bfill()
test_df['days_to_shops_closed'] = (test_df['next_shops_closed_date'] - test_df['date']).dt.days
test_df.drop(columns=['next_shops_closed_date'], inplace=True)

# ✅ 3. Mark the day after the store was closed
test_df['day_after_closing'] = (
    (test_df['shops_closed'] == 0) & (test_df['shops_closed'].shift(1) == 1)
).astype(int)

# ✅ 4. Mark long weekends (consecutive shop closures)
test_df['long_weekend'] = (
    (test_df['shops_closed'] == 1) & (test_df['shops_closed'].shift(1) == 1)
).astype(int)

# ✅ Final check
print("Columns in train_df after feature engineering:", train_df.columns.tolist())
print("Columns in test_df after feature engineering:", test_df.columns.tolist())


# In[29]:


# Define lag_features
lag_features = [1, 7, 14, 28]  # Lag of 1, 7, 14, and 28 days

# Ensure the 'date' column is in datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Group the data by unique_id and warehouse
train_grouped = train_df.groupby(["unique_id", "warehouse"])

# Calculate lag features for train_df
for i in lag_features:
    # Create lag column
    train_df[f"sales_item_warehouse_lag_{i}"] = train_grouped["sales"].shift(i)
    # Fill NaN with the last value of 'sales' in the group, if not available then fill with 0
    train_df[f"sales_item_warehouse_lag_{i}"] = train_df[f"sales_item_warehouse_lag_{i}"].fillna(
        train_grouped["sales"].transform("last")
    ).fillna(0)

# Check for NaN values in the lag columns of train_df
print("\nCheck for NaN values in lag columns of train_df:")
print(train_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].isna().sum())

# Show statistical summary for the lag columns in train_df
print("\nDescriptive statistics for lag columns in train_df:")
print(train_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].describe())

# --- Processing test_df ---

# Group the data by unique_id and warehouse
test_grouped = test_df.groupby(["unique_id", "warehouse"])

# Calculate lag features for test_df
for i in lag_features:
    # Create lag column
    test_df[f"sales_item_warehouse_lag_{i}"] = test_grouped["sales"].shift(i)
    # Fill NaN with the last value of 'sales' in the group, if not available then fill with 0
    test_df[f"sales_item_warehouse_lag_{i}"] = test_df[f"sales_item_warehouse_lag_{i}"].fillna(
        test_grouped["sales"].transform("last")
    ).fillna(0)

# Check for NaN values in the lag columns of test_df
print("\nCheck for NaN values in lag columns of test_df:")
print(test_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].isna().sum())

# Show statistical summary for the lag columns in test_df
print("\nDescriptive statistics for lag columns in test_df:")
print(test_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].describe())


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 12))

# Compute correlation matrix only on numeric columns
corr_matrix = train_df.select_dtypes(include='number').corr().abs()

# Draw heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()


# In[31]:


columns_to_drop = ['weight', 'availability']
train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')


# In[32]:


from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# ✅ Optimize data types to reduce memory usage
for col in train_df.select_dtypes('float64'):
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes('int64'):
    train_df[col] = train_df[col].astype('int32')
for col in train_df.select_dtypes('bool'):
    train_df[col] = train_df[col].astype('uint8')

for col in test_df.select_dtypes('float64'):
    test_df[col] = test_df[col].astype('float32')
for col in test_df.select_dtypes('int64'):
    test_df[col] = test_df[col].astype('int32')
for col in test_df.select_dtypes('bool'):
    test_df[col] = test_df[col].astype('uint8')

# ✅ Label encode categorical columns
categorical_cols = ['warehouse', 'holiday_name', 'name', 'L1_category_name_en', 
                    'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']

label_encoders = {}
for col in categorical_cols:
    if col in train_df.columns and col in test_df.columns:
        le = LabelEncoder()
        train_vals = train_df[col].astype(str).values
        test_vals = test_df[col].astype(str).values
        combined = np.concatenate([train_vals, test_vals])
        le.fit(combined)
        train_df[col] = le.transform(train_vals)
        test_df[col] = le.transform(test_vals)
        label_encoders[col] = le
        del combined

# ✅ Filter numeric columns excluding ID/date/sales/availability
excluded_cols = ["unique_id", "date", "sales", "availability"]
valid_dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8]

# Use pandas methods to ensure safe dtype access
train_features = train_df.select_dtypes(include='number').columns.difference(excluded_cols).tolist()
test_features = test_df.select_dtypes(include='number').columns.difference(excluded_cols).tolist()

# ✅ Get common features
features = list(set(train_features) & set(test_features))

# ✅ Print the feature types
print("Feature data types in train_df:")
for col in features:
    print(f"{col}: {train_df[col].dtype}")

print("\nFeature data types in test_df:")
for col in features:
    print(f"{col}: {test_df[col].dtype}")

# ✅ Fill any missing values with column-wise mean
train_df[features] = train_df[features].fillna(train_df[features].mean())
test_df[features] = test_df[features].fillna(train_df[features].mean())  # Use train_df mean

# ✅ Define training and validation periods
target = "sales"
training_dates = (pd.to_datetime('2022-01-01'), train_df["date"].max() - pd.Timedelta(days=14))
validation_dates = (training_dates[1] + pd.Timedelta(days=1), train_df["date"].max())

# ✅ Prepare training and validation datasets
X_train = train_df[train_df["date"].between(*training_dates)][features]
y_train = train_df[train_df["date"].between(*training_dates)][target]
X_val = train_df[train_df["date"].between(*validation_dates)][features]
y_val = train_df[train_df["date"].between(*validation_dates)][target]

# Get unique IDs for weight mapping
unique_id_train = train_df[train_df["date"].between(*training_dates)]["unique_id"]
unique_id_val = train_df[train_df["date"].between(*validation_dates)]["unique_id"]

# ✅ Prepare weights
weight_map = weights_df.set_index('unique_id')['weight'].to_dict()

# Assign default weight = 1.0 for any missing IDs
missing_ids = set(unique_id_val) - set(weight_map.keys())
if missing_ids:
    print(f"Missing IDs in weight_map: {missing_ids}")
    for mid in missing_ids:
        weight_map[mid] = 1.0

# Optional: Prepare weights array for validation
sample_weights_val = unique_id_val.map(weight_map).values


# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Simulate or load your dataset here
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'feature1': np.random.rand(n_samples),
    'feature2': np.random.rand(n_samples),
    'feature3': np.random.rand(n_samples)
})
y = 5 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples)

# Create fake unique_ids and sample weights
unique_id = pd.Series(np.arange(n_samples))
weights = pd.Series(np.random.rand(n_samples))
weight_map = dict(zip(unique_id, weights))

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
unique_id_train = unique_id.iloc[X_train.index]
unique_id_val = unique_id.iloc[X_val.index]
sample_weights_val = unique_id_val.map(weight_map).values

# Train XGBoost model (offline, no Optuna)
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
model.fit(
    X_train,
    y_train,
    sample_weight=unique_id_train.map(weight_map).values
)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
wmae = mean_absolute_error(y_val, y_pred, sample_weight=sample_weights_val)
r2 = r2_score(y_val, y_pred, sample_weight=sample_weights_val)
print(f"✅ Weighted MAE: {wmae:.4f}")
print(f"✅ Weighted R² Score: {r2:.4f}")

# Plot Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.3, edgecolors='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("📈 Predicted vs Actual Sales")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

# ✅ Define preprocessing pipeline for Ridge Regression
ridge_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0, random_state=42))
])

# ✅ Define LightGBM model
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='mae',
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# ✅ Define XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# ✅ Sample weights
sample_weights_train = unique_id_train.map(weight_map).values
sample_weights_val = unique_id_val.map(weight_map).values

# ✅ Fit models
#ridge_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights_train)
lgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
#xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

# ✅ Predict
#ridge_pred = ridge_pipeline.predict(X_val)
lgb_pred = lgb_model.predict(X_val)
#xgb_pred = xgb_model.predict(X_val)

# ✅ Simple average ensemble
ensemble_pred = lgb_pred 

# ✅ Evaluate with WMAE and R²
wmae = mean_absolute_error(y_val, ensemble_pred, sample_weight=sample_weights_val)
r2 = r2_score(y_val, ensemble_pred, sample_weight=sample_weights_val)

print(f"✅ Ensemble WMAE: {wmae:.4f}")
print(f"✅ Ensemble R² Score: {r2:.4f}")

# ✅ Visualize Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_val, ensemble_pred, alpha=0.3, label='Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual (Ensemble Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[35]:


from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Train base models (bagging + boosting)
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='mae',
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Fit base models
xgb_model.fit(X_train, y_train, sample_weight=unique_id_train.map(weight_map).values)
lgb_model.fit(X_train, y_train, sample_weight=unique_id_train.map(weight_map).values)
rf_model.fit(X_train, y_train)

# Step 2: Get base model predictions on validation set
xgb_pred = xgb_model.predict(X_val)
lgb_pred = lgb_model.predict(X_val)
rf_pred  = rf_model.predict(X_val)

# Step 3: Stack predictions into new feature set for meta-model
stacked_val = np.column_stack((xgb_pred, lgb_pred, rf_pred))

# Step 4: Train meta-model (Ridge Regression)
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(stacked_val, y_val)

# Step 5: Evaluate meta-model (stacking)
meta_pred = meta_model.predict(stacked_val)
sample_weights_val = unique_id_val.map(weight_map).values

wmae = mean_absolute_error(y_val, meta_pred, sample_weight=sample_weights_val)
r2 = r2_score(y_val, meta_pred, sample_weight=sample_weights_val)

print(f"✅ Stacked Ensemble WMAE: {wmae:.4f}")
print(f"✅ Stacked Ensemble R² Score: {r2:.4f}")

# Optional: Visualize predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_val, meta_pred, alpha=0.3, label='Predicted vs Actual')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("📈 Predicted vs Actual (Stacked Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

# ✅ Define preprocessing pipeline for Ridge Regression
ridge_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0, random_state=42))
])

# ✅ Define LightGBM model
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='mae',
    n_estimators=400,
    learning_rate=0.1,
    random_state=42
)

# ✅ Define XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# ✅ Sample weights
sample_weights_train = unique_id_train.map(weight_map).values
sample_weights_val = unique_id_val.map(weight_map).values

# ✅ Fit models
ridge_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights_train)
lgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

# ✅ Predict
ridge_pred = ridge_pipeline.predict(X_val)
lgb_pred = lgb_model.predict(X_val)
xgb_pred = xgb_model.predict(X_val)

# ✅ Simple average ensemble
ensemble_pred = (ridge_pred + lgb_pred + xgb_pred) / 3

# ✅ Evaluate with WMAE and R²
wmae = mean_absolute_error(y_val, ensemble_pred, sample_weight=sample_weights_val)
r2 = r2_score(y_val, ensemble_pred, sample_weight=sample_weights_val)

print(f"✅ Ensemble WMAE: {wmae:.4f}")
print(f"✅ Ensemble R² Score: {r2:.4f}")

# ✅ Visualize Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_val, ensemble_pred, alpha=0.3, label='Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual (Ensemble Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[37]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Replace this with your actual dataset
# X = features dataframe
# y = target variable (sales)
# unique_id = Series identifying groups
# weight_map = dict with weights per unique_id
# Ensure all are defined before this step

# Sample weights
sample_weights = unique_id.map(weight_map).values

# Initialize base models
xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
stacked_train = np.zeros((X.shape[0], 3))

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw_tr = sample_weights[train_idx]

    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    rf_model.fit(X_tr, y_tr)

    stacked_train[val_idx, 0] = xgb_model.predict(X_val)
    stacked_train[val_idx, 1] = lgb_model.predict(X_val)
    stacked_train[val_idx, 2] = rf_model.predict(X_val)

# Meta-model
meta_model = BayesianRidge()
meta_model.fit(stacked_train, y)

# Evaluation
meta_pred = meta_model.predict(stacked_train)
wmae = mean_absolute_error(y, meta_pred, sample_weight=sample_weights)
r2 = r2_score(y, meta_pred, sample_weight=sample_weights)

print(f"✅ Stacked Ensemble WMAE: {wmae:.4f}")
print(f"✅ Stacked Ensemble R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y, meta_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('📈 Predicted vs Actual (Stacked Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[38]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Sample weights
sample_weights = unique_id.map(weight_map).values

# Initialize base models
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=-1)
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)

# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
stacked_train = np.zeros((X.shape[0], 3))
meta_targets = np.zeros(X.shape[0])

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw_tr = sample_weights[train_idx]

    # Fit base models
    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    rf_model.fit(X_tr, y_tr)

    # Predict base models
    stacked_train[val_idx, 0] = xgb_model.predict(X_val)
    stacked_train[val_idx, 1] = lgb_model.predict(X_val)
    stacked_train[val_idx, 2] = rf_model.predict(X_val)

    meta_targets[val_idx] = y_val  # For safe matching

# Fit meta-model
meta_model = BayesianRidge()
meta_model.fit(stacked_train, meta_targets)

# Predict final stacked output
meta_pred = meta_model.predict(stacked_train)

# Evaluation
wmae = mean_absolute_error(meta_targets, meta_pred, sample_weight=sample_weights)
r2 = r2_score(meta_targets, meta_pred, sample_weight=sample_weights)

print(f"✅ Optimized Stacked Ensemble WMAE: {wmae:.4f}")
print(f"✅ Optimized Stacked Ensemble R² Score: {r2:.4f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(meta_targets, meta_pred, alpha=0.4, edgecolor='k')
plt.plot([meta_targets.min(), meta_targets.max()], [meta_targets.min(), meta_targets.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('📈 Predicted vs Actual (Optimized Stacked Model)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[39]:


# Create prediction DataFrame
solution_df = pd.DataFrame({
    'unique_id': unique_id,
    'actual_sales': meta_targets,
    'predicted_sales': meta_pred
})

# Save to CSV
solution_df.to_csv('stacked_model_predictions.csv', index=False)
print("✅ Predictions saved to 'stacked_model_predictions.csv'")


# In[40]:


from sklearn.linear_model import SGDRegressor

meta_model = SGDRegressor(loss='squared_error', penalty='l2', learning_rate='adaptive', eta0=0.01, random_state=42)
meta_model.fit(stacked_train, meta_targets, sample_weight=sample_weights)

meta_pred = meta_model.predict(stacked_train)

# Evaluate
wmae = mean_absolute_error(meta_targets, meta_pred, sample_weight=sample_weights)
r2 = r2_score(meta_targets, meta_pred, sample_weight=sample_weights)

print(f"✅ SGDRegressor Meta WMAE: {wmae:.4f}")
print(f"✅ SGDRegressor Meta R² Score: {r2:.4f}")


# In[ ]:





# In[41]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

# Sample weights
sample_weights = unique_id.map(weight_map).values

# Initialize base models
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=-1)
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
cat_model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, random_seed=42,
                              verbose=0, task_type="CPU", loss_function='MAE')

# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
stacked_train = np.zeros((X.shape[0], 4))  # now 4 base models
meta_targets = np.zeros(X.shape[0])

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw_tr = sample_weights[train_idx]

    # Fit base models
    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    rf_model.fit(X_tr, y_tr)
    cat_model.fit(X_tr, y_tr, sample_weight=sw_tr)

    # Predict base models
    stacked_train[val_idx, 0] = xgb_model.predict(X_val)
    stacked_train[val_idx, 1] = lgb_model.predict(X_val)
    stacked_train[val_idx, 2] = rf_model.predict(X_val)
    stacked_train[val_idx, 3] = cat_model.predict(X_val)

    meta_targets[val_idx] = y_val  # For safe matching

# Fit meta-model
meta_model = BayesianRidge()
meta_model.fit(stacked_train, meta_targets)

# Predict final stacked output
meta_pred = meta_model.predict(stacked_train)

# Evaluation
wmae = mean_absolute_error(meta_targets, meta_pred, sample_weight=sample_weights)
r2 = r2_score(meta_targets, meta_pred, sample_weight=sample_weights)

print(f"✅ CatBoost-Enhanced Stacked Ensemble WMAE: {wmae:.4f}")
print(f"✅ CatBoost-Enhanced Stacked Ensemble R² Score: {r2:.4f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(meta_targets, meta_pred, alpha=0.4, edgecolor='k')
plt.plot([meta_targets.min(), meta_targets.max()], [meta_targets.min(), meta_targets.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('📈 Predicted vs Actual (CatBoost-Enhanced Stacked Model)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[42]:


import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

models = {
    'Ridge': {
        'model': Ridge(random_state=43),
        'params': {
            'model__alpha': [0.1, 1, 10, 100],
            'model__solver': ['auto', 'lsqr']
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'model__n_estimators': [100],
            'model__max_depth': [10, None]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        'params': {
            'model__learning_rate': [0.05],
            'model__n_estimators': [100],
            'model__num_leaves': [31]
        }
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
        'params': {
            'model__learning_rate': [0.05],
            'model__n_estimators': [100],
            'model__max_depth': [6]
        }
    }
}

results = {}
best_model = None
best_score = float('inf')

for name, cfg in models.items():
    print(f"\n🚀 Training {name}...")
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', cfg['model'])
    ])
    
    gs = GridSearchCV(pipe, cfg['params'], cv=3, scoring='neg_mean_absolute_error', verbose=1)
    
    # Fit with sample weights
    gs.fit(X_train, y_train, model__sample_weight=unique_id_train.map(weight_map).values)
    
    # Predict
    y_pred = gs.predict(X_val)
    
    # Metrics
    mae = mean_absolute_error(y_val, y_pred, sample_weight=sample_weights_val)
    mse = mean_squared_error(y_val, y_pred, sample_weight=sample_weights_val)
    r2 = r2_score(y_val, y_pred, sample_weight=sample_weights_val)
    
    results[name] = {
        'model': gs.best_estimator_,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'params': gs.best_params_
    }
    
    print(f"✅ {name} | WMAE: {mae:.4f} | R²: {r2:.4f}")
    
    # Save best model
    if mae < best_score:
        best_score = mae
        best_model = gs.best_estimator_
        best_model_name = name

# ✅ Save best model to disk
with open(f"best_model_{best_model_name}.pkl", 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n🏆 Best Model: {best_model_name} with WMAE = {best_score:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:




