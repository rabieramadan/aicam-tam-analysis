import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
import os
import re

# Set plot style for publication
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def clean_likert(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r'^\d', str(val).strip())
    if match:
        return float(match.group())
    return np.nan

def load_data(filepath):
    try:
        df_raw = pd.read_csv(filepath, header=None)
        # Find header rows (we assume headers are scattered in the first 2-3 rows)
        headers = []
        for i in range(df_raw.shape[1]):
            # Combine the first 3 rows to form a comprehensive header name
            col_text = " ".join(df_raw.iloc[0:2, i].dropna().astype(str))
            # Clean up the header a bit
            col_text = re.sub(r'\(.*?\)', '', col_text).strip()
            if not col_text:
                col_text = f"Column_{i}"
            headers.append(col_text[:60])
            
        # Ensure unique headers
        final_headers = []
        seen = {}
        for h in headers:
            if h in seen:
                seen[h] += 1
                final_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                final_headers.append(h)
                
        df = pd.read_csv(filepath, header=1) # Load actual data
        df.columns = final_headers
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def generate_table1_demographics(df, output_dir):
    print("Generating Table 1: Demographics...")
    # Assume first few columns are Timestamp, Gender, Year of Study, etc.
    demog_cols = df.columns[1:4] # Adjust based on data
    tables = []
    
    for col in demog_cols:
        counts = df[col].value_counts()
        percs = df[col].value_counts(normalize=True) * 100
        temp = pd.DataFrame({'N': counts, '%': percs.round(1)})
        temp.index.name = col
        tables.append(temp)
        
    res = pd.concat(tables, keys=[c for c in demog_cols])
    res.to_csv(os.path.join(output_dir, 'Table1_Demographics.csv'))
    return res

def generate_table2_descriptives(df, output_dir):
    print("Generating Table 2: Descriptive Stats...")
    # Find likert columns
    likert_df = pd.DataFrame()
    for col in df.columns:
        non_nulls = df[col].dropna().astype(str)
        if len(non_nulls) == 0: continue
        if non_nulls.str.match(r'^\d').mean() > 0.2:
            likert_df[col] = df[col].apply(clean_likert)
            
    stats = pd.DataFrame({
        'Mean': likert_df.mean(),
        'SD': likert_df.std(),
        'Skewness': likert_df.skew(),
        'Kurtosis': likert_df.kurt()
    }).round(2)
    
    stats.to_csv(os.path.join(output_dir, 'Table2_Descriptives.csv'))
    return likert_df

def generate_figure1_tools(df, output_dir):
    print("Generating Figure 1: AI Tools Utilized...")
    # Find tools column
    tools_col = None
    for col in df.columns:
        if 'Which AI tools' in col:
            tools_col = col
            break
            
    if not tools_col: return
    
    # Tools are often semicolon separated
    tools_series = df[tools_col].dropna().astype(str)
    all_tools = []
    for t in tools_series:
        all_tools.extend([x.strip() for x in t.split(';')])
        
    tool_counts = pd.Series(all_tools).value_counts().head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tool_counts.values, y=tool_counts.index, palette='viridis')
    plt.title('Top AI Tools Utilized by Literature Students')
    plt.xlabel('Number of Students')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure1_AITools.png'), dpi=300)
    plt.close()

def generate_figure2_benefits(likert_df, output_dir):
    print("Generating Figure 2: Perceived Benefits vs Challenges...")
    # Pick a few key benefits columns
    benefits_cols = [c for c in likert_df.columns if 'benefit' in c.lower() or 'agree' in c.lower()][:8]
    if not benefits_cols: return
    
    # For a diverging bar chart, we usually need counts of 1,2,3,4,5
    counts = {}
    for col in benefits_cols:
        counts[col[:40]] = likert_df[col].value_counts().reindex([1,2,3,4,5], fill_value=0)
        
    counts_df = pd.DataFrame(counts).T
    # Calculate percentages
    counts_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100
    
    # Plotting stacked bar
    counts_df.plot(kind='barh', stacked=True, colormap='RdYlGn', figsize=(12, 8))
    plt.legend(['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Perceived Benefits and Attitudes')
    plt.xlabel('Percentage of Responses')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure2_Benefits.png'), dpi=300)
    plt.close()

def generate_figure3_clustering(likert_df, output_dir):
    print("Generating Figure 3: Dendrogram (HAC)...")
    df_clean = likert_df.fillna(likert_df.median())
    df_clean = df_clean.dropna(axis=1, how='all')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    plt.figure(figsize=(12, 6))
    Z = linkage(X_scaled, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (Student Profiles)')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure3_Dendrogram.png'), dpi=300)
    plt.close()

def generate_figure4_rf(likert_df, output_dir):
    print("Generating Figure 4 & Table 3: Random Forest...")
    # Find a target column (e.g., Ethical concerns or Engagement)
    target_col = None
    for col in likert_df.columns:
        if 'concern' in col.lower() or 'impact' in col.lower():
            target_col = col
            break
            
    if not target_col: return
    
    y = likert_df[target_col].fillna(likert_df[target_col].median())
    X = likert_df.drop(columns=[target_col]).fillna(likert_df.drop(columns=[target_col]).median())
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [X.columns[i][:50] for i in indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=top_features, palette='mako')
    plt.title(f'Top Predictors for: {target_col[:40]}')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure4_FeatureImportance.png'), dpi=300)
    plt.close()
    
    # Table 3
    y_pred = rf.predict(X)
    metrics = pd.DataFrame({
        'Metric': ['MSE', 'R-squared (R2)'],
        'Value': [mean_squared_error(y, y_pred), r2_score(y, y_pred)]
    })
    metrics.to_csv(os.path.join(output_dir, 'Table3_RF_Metrics.csv'), index=False)

def generate_table4_apriori(df, output_dir):
    print("Generating Table 4: Apriori Association Rules...")
    # Convert dataframe to boolean for apriori
    # We will pick a few categorical columns
    cat_df = pd.DataFrame()
    for col in df.columns[:15]: # Take first few to avoid exploding memory
        if df[col].nunique() < 10:
            dummies = pd.get_dummies(df[col], prefix=col[:10])
            cat_df = pd.concat([cat_df, dummies], axis=1)
            
    cat_df = cat_df.astype(bool)
    
    try:
        frequent_itemsets = apriori(cat_df, min_support=0.15, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        rules = rules.sort_values('lift', ascending=False).head(10)
        
        # Clean up rules for table
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_out = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
        rules_out.to_csv(os.path.join(output_dir, 'Table4_AssociationRules.csv'), index=False)
    except Exception as e:
        print(f"Apriori failed (often happens if data is too sparse or mlxtend issue): {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "XLS data.csv")
    
    df = load_data(data_path)
    if df is None: return
    
    generate_table1_demographics(df, script_dir)
    likert_df = generate_table2_descriptives(df, script_dir)
    generate_figure1_tools(df, script_dir)
    if likert_df is not None and not likert_df.empty:
        generate_figure2_benefits(likert_df, script_dir)
        generate_figure3_clustering(likert_df, script_dir)
        generate_figure4_rf(likert_df, script_dir)
    generate_table4_apriori(df, script_dir)
    
    print("\nAll assets generated successfully in", script_dir)

if __name__ == '__main__':
    main()
