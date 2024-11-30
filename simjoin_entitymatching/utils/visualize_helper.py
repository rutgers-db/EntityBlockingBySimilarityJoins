# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_semantic_distribution(is_fp):
    tab = pd.read_csv("../../output/debug/false_positive_second.csv") if is_fp else pd.read_csv("../../output/debug/true_positive_second.csv")
    
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create a histogram with a KDE overlay
    sns.histplot(tab['cosine'], bins=20, kde=True, color='blue', stat='probability')

    # Add labels and title
    plt.title('Distribution of Cosine Values', fontsize=16)
    plt.xlabel('Cosine', fontsize=14)
    plt.ylabel('Proba', fontsize=14)

    # Show the plot
    if is_fp:
        plt.savefig('../../output/debug/figs/fp_semantic.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("../../output/debug/figs/tp_semantic.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def show_first_proba_distribution(is_fp):
    tab = pd.read_csv("../../output/debug/false_positive_second.csv") if is_fp else pd.read_csv("../../output/debug/true_positive_second.csv")
    
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create a histogram with a KDE overlay
    sns.histplot(tab['first proba'], bins=20, kde=True, color='blue', stat='probability')

    # Add labels and title
    plt.title('Distribution of first proba', fontsize=16)
    plt.xlabel('first proba', fontsize=14)
    plt.ylabel('Proba', fontsize=14)

    # Show the plot
    if is_fp:
        plt.savefig('../../output/debug/figs/fp_first_proba.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("../../output/debug/figs/tp_first_proba.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def show_second_proba_distribution(is_fp):
    tab = pd.read_csv("../../output/debug/false_positive_second.csv") if is_fp else pd.read_csv("../../output/debug/true_positive_second.csv")
    
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create a histogram with a KDE overlay
    sns.histplot(tab['second proba'], bins=20, kde=True, color='blue', stat='probability')

    # Add labels and title
    plt.title('Distribution of second proba', fontsize=16)
    plt.xlabel('second proba', fontsize=14)
    plt.ylabel('Proba', fontsize=14)

    # Show the plot
    if is_fp:
        plt.savefig('../../output/debug/figs/fp_second_proba.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("../../output/debug/figs/tp_second_proba.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def show_diff_proba_distribution(is_fp, is_corre):
    tab = pd.read_csv("../../output/debug/false_positive_second.csv") if is_fp else pd.read_csv("../../output/debug/true_positive_second.csv")
    tab["diff proba"] = tab["second proba"] - tab["first proba"]
    
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    if not is_corre:
        # Create a histogram with a KDE overlay
        sns.histplot(tab['diff proba'], bins=20, kde=True, color='blue', stat='probability')

        # Add labels and title
        plt.title('Distribution of diff proba', fontsize=16)
        plt.xlabel('diff proba', fontsize=14)
        plt.ylabel('Proba', fontsize=14)
    else:
        sns.scatterplot(data=tab, x='diff proba', y='first proba', color='blue')

        # Add a regression line to visualize the trend
        sns.regplot(data=tab, x='diff proba', y='first proba', scatter=False, color='red')

        # Add labels and title
        plt.title('Relationship between diff proba and first proba', fontsize=16)
        plt.xlabel('diff proba', fontsize=14)
        plt.ylabel('first proba', fontsize=14)

    # Show the plot
    if is_fp:
        plt.savefig('../../output/debug/figs/fp_diff_proba.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("../../output/debug/figs/tp_diff_proba.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def show_relation_first_proba_semantic(is_fp):
    tab = pd.read_csv("../../output/debug/false_positive_second.csv") if is_fp else pd.read_csv("../../output/debug/true_positive_second.csv")
    tab["diff proba"] = tab["second proba"] - tab["first proba"]

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tab, x='cosine', y='diff proba', color='blue')

    # Add a regression line to visualize the trend
    sns.regplot(data=tab, x='cosine', y='diff proba', scatter=False, color='red')

    # Add labels and title
    plt.title('Relationship between Cosine and First Probability', fontsize=16)
    plt.xlabel('Cosine', fontsize=14)
    plt.ylabel('First Probability', fontsize=14)

     # Show the plot
    if is_fp:
        plt.savefig('../../output/debug/figs/fp_first_proba_semantic.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig("../../output/debug/figs/tp_first_proba_semantic.png", dpi=300, bbox_inches='tight')
    plt.show()