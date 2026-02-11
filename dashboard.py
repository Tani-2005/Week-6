import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Setup
sns.set_theme(style="whitegrid", palette="Set2")
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Load Data
df = pd.read_csv("sales_data.csv")
df.columns = df.columns.str.strip()

# Auto-detect important columns
sales_col = next((c for c in df.columns if "sale" in c.lower() or "revenue" in c.lower()), None)
category_col = next((c for c in df.columns if "category" in c.lower()), None)
region_col = next((c for c in df.columns if "region" in c.lower()), None)
date_col = next((c for c in df.columns if "date" in c.lower()), None)

# Fallback to Product if Category missing
if not category_col:
    category_col = next((c for c in df.columns if "product" in c.lower()), None)

# Date processing
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = df[date_col].dt.month
    df["Year"] = df[date_col].dt.year

df = df.dropna()

# Aggregations
monthly_sales = df.groupby("Month")[sales_col].sum().reset_index()
category_sales = df.groupby(category_col)[sales_col].sum().reset_index()
region_sales = df.groupby(region_col)[sales_col].sum().reset_index()

# Create 3x2 Subplot Grid
fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle("INTERACTIVE SALES PERFORMANCE DASHBOARD",
             fontsize=26, fontweight="bold", y=0.98)

# 1️⃣ Line Plot
sns.lineplot(data=monthly_sales,
             x="Month",
             y=sales_col,
             marker="o",
             ax=axes[0,0])
axes[0,0].set_title("Monthly Sales Trend", fontsize=14)

# 2️⃣ Bar Plot
sns.barplot(data=category_sales,
            x=category_col,
            y=sales_col,
            ax=axes[0,1])
axes[0,1].set_title("Sales by Category", fontsize=14)
axes[0,1].tick_params(axis='x', rotation=45)

# 3️⃣ Pie Chart (matplotlib)
axes[1,0].pie(region_sales[sales_col],
              labels=region_sales[region_col],
              autopct='%1.1f%%')
axes[1,0].set_title("Sales by Region", fontsize=14)

# 4️⃣ Scatter Plot
sns.scatterplot(data=df,
                x="Month",
                y=sales_col,
                ax=axes[1,1])
axes[1,1].set_title("Sales Scatter Distribution", fontsize=14)

# 5️⃣ Box Plot
sns.boxplot(data=df,
            y=sales_col,
            ax=axes[2,0])
axes[2,0].set_title("Sales Distribution (Box Plot)", fontsize=14)

# 6️⃣ Heatmap
corr = df.select_dtypes(include=["int64","float64"]).corr()
sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            ax=axes[2,1])
axes[2,1].set_title("Correlation Heatmap", fontsize=14)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save dashboard
plt.savefig("visualizations/seaborn_sales_dashboard.png", dpi=300)
plt.show()

print("✅ 6-Chart Seaborn Dashboard Created Successfully!")
