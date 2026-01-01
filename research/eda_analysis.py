import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
DATA_PATH = "research/Job_Placement_Data_Enhanced.csv"
OUTPUT_DIR = "research/eda_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    """Loads the dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} was not found.")
    return pd.read_csv(path)

def generate_boxplots(df, numerical_cols, target_col):
    """Generates boxplots for numerical columns vs target column."""
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'Boxplot of {col} by {target_col}')
        plt.savefig(f"{OUTPUT_DIR}/boxplot_{col}.png")
        print(f"Saved boxplot for {col} to {OUTPUT_DIR}/boxplot_{col}.png")
        plt.close()

def detect_outliers(df, numerical_cols):
    """Detects outliers using IQR method."""
    outliers_report = {}
    
    print("\n--- Outlier Detection (IQR Method) ---")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            outliers_report[col] = {
                'count': len(outliers),
                'bounds': (lower_bound, upper_bound),
                'indices': outliers.index.tolist(),
                'values': outliers[col].tolist()
            }
            print(f"\nFeature: {col}")
            print(f"  IQR: {IQR:.2f} (Q1={Q1:.2f}, Q3={Q3:.2f})")
            print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Number of Outliers: {len(outliers)}")
            print(f"  Outlier Indices: {outliers.index.tolist()}")
            # print(f"  Outlier Values: {outliers[col].tolist()}")
        else:
            print(f"\nFeature: {col} - No outliers detected.")
            
    return outliers_report

def main():
    try:
        # Load Data
        df = load_data(DATA_PATH)
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        
        # Identify numerical columns
        # Excluding 'ssc_board', 'hsc_board', 'hsc_subject', 'undergrad_degree', 'work_experience', 'specialisation', 'status', 'gender'
        # based on typical categorical nature. Inspecting dtypes is safer.
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove ID column if it exists (usually not useful for analysis)
        if 'student_id' in numerical_cols:
            numerical_cols.remove('student_id')
            
        print(f"Numerical columns found: {numerical_cols}")
        
        if 'status' not in df.columns:
            print("Error: 'status' column not found in dataset.")
            return

        # Generate Boxplots
        generate_boxplots(df, numerical_cols, 'status')
        
        # Detect Outliers
        detect_outliers(df, numerical_cols)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
