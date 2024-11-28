# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_semantic_distribution(is_fp):
    tab = pd.read_csv("output/debug/false_positive_second.csv") if is_fp else pd.read_csv("output/debug/true_positive_second.csv")
    
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create a histogram with a KDE overlay
    sns.histplot(tab['cosine'], bins=20, kde=True, color='blue', stat='density')

    # Add labels and title
    plt.title('Distribution of Cosine Values', fontsize=16)
    plt.xlabel('Cosine', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # Show the plot
    if is_fp:
        plt.savefig('output/debug/figs/fp_semantic.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("output/debug/figs/tp_semantic.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    
def show_relation_first_proba_semantic(is_fp):
    tab = pd.read_csv("output/debug/false_positive_second.csv") if is_fp else pd.read_csv("output/debug/true_positive_second.csv")

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tab, x='cosine', y='first proba', color='blue')

    # Add a regression line to visualize the trend
    sns.regplot(data=tab, x='cosine', y='first proba', scatter=False, color='red')

    # Add labels and title
    plt.title('Relationship between Cosine and First Probability', fontsize=16)
    plt.xlabel('Cosine', fontsize=14)
    plt.ylabel('First Probability', fontsize=14)

     # Show the plot
    if is_fp:
        plt.savefig('output/debug/figs/fp_first_proba_semantic.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("output/debug/figs/tp_first_proba_semantic.png", dpi=300, bbox_inches='tight')
    # plt.show()