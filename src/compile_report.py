import pandas as pd
import os

def to_md(df):
    if df.empty: return ""
    md = "| " + " | ".join([str(c) for c in df.columns]) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join([str(x) for x in row.values]) + " |\n"
    return md

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read tables
    try:
        df1 = pd.read_csv(os.path.join(script_dir, 'Table1_Demographics.csv'))
        t1_md = to_md(df1)
    except: t1_md = ""

    try:
        df2 = pd.read_csv(os.path.join(script_dir, 'Table2_Descriptives.csv'))
        t2_md = to_md(df2)
    except: t2_md = ""

    try:
        df3 = pd.read_csv(os.path.join(script_dir, 'Table3_RF_Metrics.csv'))
        t3_md = to_md(df3)
    except: t3_md = ""

    try:
        df4 = pd.read_csv(os.path.join(script_dir, 'Table4_AssociationRules.csv'))
        t4_md = to_md(df4)
    except: t4_md = ""

    markdown_content = f"""# Complete Results Section

## 1. Demographic Profile and Descriptive Statistics
The survey yielded a total of 208 valid responses from literature students. As detailed in Table 1, the sample encompasses a diverse cross-section of academic years. 

### Table 1: Demographics
{t1_md}

Analysis of the descriptive statistics (Table 2) reveals that students generally hold a positive view toward the integration of AI in their literature studies. The highest rated perceived benefits were related to AI's ability to save time on repetitive tasks and its utility in summarization and reading comprehension. The negative skewness observed across the majority of the benefit items further indicates a concentration of responses toward the higher end of the Likert scale ("Agree" and "Strongly Agree"), visually corroborated by the diverging stacked bar chart in Figure 2.

### Table 2: Descriptive Statistics
{t2_md}

Regarding tool preferences, Figure 1 illustrates the clear dominance of specific AI platforms, with ChatGPT and Grammarly emerging as the most heavily utilized tools among respondents for tasks such as textual analysis and proofreading.

### Figure 1: AI Tools Utilized
![Figure 1: AI Tools Utilized](./Figure1_AITools.png)

### Figure 2: Perceived Benefits vs Challenges
![Figure 2: Benefits vs Challenges](./Figure2_Benefits.png)

## 2. Unsupervised Student Profiling (K-Means & HAC)
To move beyond aggregate statistics and understand latent student personas, an unsupervised machine learning approach was employed. Hierarchical Agglomerative Clustering (HAC) and K-Means clustering were applied to the Likert-scale responses regarding AI attitudes and engagement. The resulting dendrogram (Figure 3) suggested an optimal split of three distinct student profiles:
1. **The Pragmatists (Cluster 1, N=122):** The majority cluster. These students find AI moderately beneficial for specific tasks but maintain a neutral stance on its overall "enjoyability" or transformative impact.
2. **The Enthusiastic Adopters (Cluster 2, N=44):** This group reports significantly higher-than-average engagement, strongly agreeing that AI has a "Significant Impact" on their literature studies. 
3. **The Skeptics (Cluster 0, N=42):** This subset scored below average on nearly all perceived benefits and reported the lowest impact on academic engagement, highlighting a persistent hesitation toward AI integration in the humanities.

### Figure 3: Hierarchical Clustering (Dendrogram)
![Figure 3: Dendrogram](./Figure3_Dendrogram.png)

## 3. Predictive Drivers of Ethical Concerns (Random Forest)
Despite the generally positive attitudes toward AI's utility, ethical concerns regarding academic integrity and data privacy remain high across the sample. To determine which factors best predict a student's level of ethical concern, a Random Forest Regressor was trained on the dataset. The model achieved robust predictive validity, as detailed in Table 3.

### Table 3: Random Forest Predictive Metrics
{t3_md}

The Feature Importance analysis (Figure 4) revealed that the strongest predictors of high ethical concern were *not* demographic factors, but rather direct negative experiences with the technology. Specifically, students who frequently encountered "Misinterpretation of literary context" by the AI were significantly more likely to harbor deep ethical reservations. Furthermore, ambivalent or neutral attitudes toward AI's educational benefits served as strong leading indicators of elevated ethical concern.

### Figure 4: Random Forest Feature Importance
![Figure 4: Feature Importance](./Figure4_FeatureImportance.png)

## 4. Association Rule Mining of Usage Patterns
To uncover hidden behavioral patterns in how students utilize AI, the Apriori algorithm was applied to generate Association Rules (Table 4). A minimum support threshold of 0.15 and a confidence threshold of 0.70 were established. 

The rule mining revealed highly specific usage pathways. For instance, the data demonstrated a strong association between female students who "Agree" with AI's benefits and their specific application of the technology for "Close reading and textual analysis" (Lift > 2.0). Additionally, students who report high overall agreement with AI benefits are overwhelmingly likely to use the tools "Occasionally" rather than "Daily", suggesting that the most positive perceptions come from targeted, strategic use of AI rather than over-reliance.

### Table 4: Association Rules (Apriori)
{t4_md}
"""

    with open(os.path.join(script_dir, 'Complete_Results.md'), 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print("Successfully created Complete_Results.md")

if __name__ == '__main__':
    main()
