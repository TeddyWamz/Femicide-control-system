# Data Inspection and Cleaning Script
# Helps identify and fix data quality issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def inspect_dataset(file_path):
    """
    Comprehensive dataset inspection
    """
    print("="*70)
    print("DATASET INSPECTION REPORT")
    print("="*70)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"\n1. BASIC INFORMATION")
    print("-" * 70)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    print(f"\n2. MISSING VALUES")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values ✓")
    
    # Class distribution
    print(f"\n3. CLASS DISTRIBUTION")
    print("-" * 70)
    category_counts = df['type'].value_counts()
    print(category_counts)
    print(f"\nImbalance ratio (max/min): {category_counts.max() / category_counts.min():.2f}x")
    
    # Visualize distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    category_counts.plot(kind='bar', color='steelblue')
    plt.title('Type Distribution', fontsize=14)
    plt.xlabel('Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Type Percentage', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved class_distribution.png")
    
    # Text length analysis
    print(f"\n4. TEXT LENGTH ANALYSIS")
    print("-" * 70)
    df['text_length'] = df['tweet'].astype(str).str.len()
    df['word_count'] = df['tweet'].astype(str).str.split().str.len()
    
    print(f"Min length: {df['text_length'].min()} characters")
    print(f"Max length: {df['text_length'].max()} characters")
    print(f"Mean length: {df['text_length'].mean():.2f} characters")
    print(f"Median length: {df['text_length'].median():.2f} characters")
    
    print(f"\nMin words: {df['word_count'].min()}")
    print(f"Max words: {df['word_count'].max()}")
    print(f"Mean words: {df['word_count'].mean():.2f}")
    
    # Find very short texts
    short_texts = df[df['text_length'] < 15]
    if len(short_texts) > 0:
        print(f"\n⚠️  WARNING: {len(short_texts)} texts are very short (<15 chars)")
        print("Sample short texts:")
    print(short_texts[['tweet', 'type']].head())
    
    # Text length by category
    print(f"\n5. TEXT LENGTH BY CATEGORY")
    print("-" * 70)
    for category in df['type'].unique():
        cat_data = df[df['type'] == category]
        print(f"{category}: {cat_data['text_length'].mean():.1f} chars (avg)")
    
    # Duplicate detection
    print(f"\n6. DUPLICATE DETECTION")
    print("-" * 70)
    duplicates = df[df.duplicated(subset=['tweet'], keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  Found {len(duplicates)} duplicate texts")
        print("\nSample duplicates:")
        print(duplicates[['tweet', 'type']].head(10))
    else:
        print("No duplicates found ✓")
    
    # Sample texts per category
    print(f"\n7. SAMPLE TEXTS PER CATEGORY")
    print("-" * 70)
    for category in df['type'].unique():
        print(f"\n{category.upper()}:")
        samples = df[df['type'] == category].sample(min(3, len(df[df['type'] == category])))
        for idx, row in samples.iterrows():
            print(f"  - {row['tweet'][:100]}...")
    
    # Check for mislabeled data
    print(f"\n8. POTENTIAL MISLABELING CHECK")
    print("-" * 70)
    
    # Physical violence keywords
    physical_keywords = ['hit', 'beat', 'punch', 'kick', 'slap', 'choke', 'stab', 
                         'piga', 'chapa', 'uma', 'ruka']
    
    # Sexual violence keywords
    sexual_keywords = ['rape', 'sexual', 'assault', 'molest', 'harassment', 'grope',
                       'baka', 'lazima', 'ngono', 'ubakaji']
    
    # Emotional keywords
    emotional_keywords = ['insult', 'belittle', 'humiliate', 'threaten', 'control',
                          'tukana', 'dharau', 'shutumu', 'chukiza']
    
    potential_issues = []
    
    for idx, row in df.iterrows():
        text_lower = row['tweet'].lower()
        category = row['type']
        
        # Check physical violence
        if any(kw in text_lower for kw in physical_keywords):
            if 'physical' not in category.lower():
                potential_issues.append({
                    'text': row['tweet'][:80],
                    'current_category': category,
                    'suggested_category': 'Physical_violence',
                    'reason': 'Contains physical violence keywords'
                })
        
        # Check sexual violence
        elif any(kw in text_lower for kw in sexual_keywords):
            if 'sexual' not in category.lower():
                potential_issues.append({
                    'text': row['tweet'][:80],
                    'current_category': category,
                    'suggested_category': 'sexual_violence',
                    'reason': 'Contains sexual violence keywords'
                })
        
        # Check emotional
        elif any(kw in text_lower for kw in emotional_keywords):
            if 'emotional' not in category.lower():
                potential_issues.append({
                    'text': row['tweet'][:80],
                    'current_category': category,
                    'suggested_category': 'emotional_violence',
                    'reason': 'Contains emotional violence keywords'
                })
    
    if potential_issues:
        print(f"⚠️  Found {len(potential_issues)} potentially mislabeled samples")
        print("\nTop 10 suspicious cases:")
        for i, issue in enumerate(potential_issues[:10], 1):
            print(f"\n{i}. Text: {issue['text']}...")
            print(f"   Current: {issue['current_category']}")
            print(f"   Suggested: {issue['suggested_category']}")
            print(f"   Reason: {issue['reason']}")
        
        # Save to file
        pd.DataFrame(potential_issues).to_csv('potential_mislabeled.csv', index=False)
        print(f"\n✓ Full list saved to 'potential_mislabeled.csv'")
    else:
        print("No obvious mislabeling detected ✓")
    
    # Recommendations
    print(f"\n9. RECOMMENDATIONS")
    print("=" * 70)
    
    if category_counts.max() / category_counts.min() > 10:
        print("⚠️  SEVERE CLASS IMBALANCE")
        print("   → Use 'oversample' balancing method")
        print("   → Apply class weights during training")
    
    if len(short_texts) > 50:
        print("⚠️  MANY SHORT TEXTS")
        print("   → Remove texts with <10 characters")
        print("   → Check data quality")
    
    if len(duplicates) > 100:
        print("⚠️  MANY DUPLICATES")
        print("   → Remove duplicates before training")
    
    if len(potential_issues) > 50:
        print("⚠️  POTENTIAL LABELING ISSUES")
        print("   → Review and fix mislabeled data")
        print("   → Consider manual data cleaning")
    
    print("\n" + "="*70)
    
    return df, potential_issues

def clean_dataset(df, remove_duplicates=True, min_length=10, fix_labels=False):
    """
    Clean the dataset based on inspection findings
    """
    print("\n" + "="*70)
    print("CLEANING DATASET")
    print("="*70)
    
    original_size = len(df)
    
    # 1. Remove very short texts
    print(f"\n1. Removing texts shorter than {min_length} characters...")
    df = df[df['tweet'].astype(str).str.len() >= min_length]
    print(f"   Removed: {original_size - len(df)} samples")
    
    # 2. Remove duplicates
    if remove_duplicates:
        print(f"\n2. Removing duplicate texts...")
        before = len(df)
        df = df.drop_duplicates(subset=['tweet'], keep='first')
        print(f"   Removed: {before - len(df)} duplicates")
    
    # 3. Clean text
    print(f"\n3. Cleaning text...")
    df['tweet'] = df['tweet'].astype(str).str.strip()
    df['tweet'] = df['tweet'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
    df['tweet'] = df['tweet'].str.replace(r'http\S+', '', regex=True)  # URLs
    df['tweet'] = df['tweet'].str.replace(r'@\w+', '', regex=True)  # Mentions
    
    # 4. Standardize category names
    print(f"\n4. Standardizing category names...")
    category_mapping = {
        'physical violence': 'Physical_violence',
        'Physical violence': 'Physical_violence',
        'PHYSICAL VIOLENCE': 'Physical_violence',
        'sexual violence': 'sexual_violence',
        'Sexual violence': 'sexual_violence',
        'SEXUAL VIOLENCE': 'sexual_violence',
        'emotional violence': 'emotional_violence',
        'Emotional violence': 'emotional_violence',
        'EMOTIONAL VIOLENCE': 'emotional_violence',
        'economic violence': 'economic_violence',
        'Economic violence': 'economic_violence',
        'ECONOMIC VIOLENCE': 'economic_violence',
    }
    df['type'] = df['type'].replace(category_mapping)
    
    print(f"\n✓ Cleaned dataset size: {len(df)}")
    print(f"✓ Removed: {original_size - len(df)} samples total")
    
    return df

def create_manual_swahili_dataset():
    """
    Create high-quality manual Swahili examples for each category
    """
    manual_data = [
        # Physical Violence - Swahili
        {"text": "Alinipiga ngumi usoni na kuniumiza sana", "category": "Physical_violence"},
        {"text": "Mume wangu ananichapa kila siku na kunivunja mifupa", "category": "Physical_violence"},
        {"text": "Amenipiga mateke tumboni nikiwa mjamzito", "category": "Physical_violence"},
        {"text": "Alinisukuma kwa nguvu na kuniangusha chini", "category": "Physical_violence"},
        {"text": "Ananichapa kwa mkono na kamba", "category": "Physical_violence"},
        {"text": "Alinikwaruza shingoni na kujaribu kuninyonga", "category": "Physical_violence"},
        {"text": "Amenivunja mkono na miguu yangu inaumwa", "category": "Physical_violence"},
        {"text": "Alinipiga kibiriti uso wangu", "category": "Physical_violence"},
        
        # Physical Violence - English
        {"text": "He punched me in the face repeatedly", "category": "Physical_violence"},
        {"text": "My husband beats me every day with his fists", "category": "Physical_violence"},
        {"text": "He kicked me in the stomach while I was pregnant", "category": "Physical_violence"},
        {"text": "He pushed me down the stairs violently", "category": "Physical_violence"},
        {"text": "He choked me until I couldn't breathe", "category": "Physical_violence"},
        {"text": "He broke my arm and ribs during the attack", "category": "Physical_violence"},
        {"text": "He slapped me across the face multiple times", "category": "Physical_violence"},
        
        # Sexual Violence - Swahili
        {"text": "Alinilazimisha kufanya ngono bila ridhaa yangu", "category": "sexual_violence"},
        {"text": "Amenishikilia na kunibaka kinyaa", "category": "sexual_violence"},
        {"text": "Alinigusa vibaya sehemu za siri bila idhini", "category": "sexual_violence"},
        {"text": "Alinilazimisha kutembeza uchi mbele yake", "category": "sexual_violence"},
        {"text": "Amenitishia kunibaka kama sitamtii", "category": "sexual_violence"},
        {"text": "Alinipiga na kunilazimisha kulala naye", "category": "sexual_violence"},
        {"text": "Amenifanya mambo ya kingono bila kupenda", "category": "sexual_violence"},
        
        # Sexual Violence - English
        {"text": "He forced me to have sex against my will", "category": "sexual_violence"},
        {"text": "He raped me when I refused his advances", "category": "sexual_violence"},
        {"text": "He touched me inappropriately without consent", "category": "sexual_violence"},
        {"text": "He sexually assaulted me repeatedly", "category": "sexual_violence"},
        {"text": "He threatened to rape me if I didn't comply", "category": "sexual_violence"},
        {"text": "He forced himself on me despite my protests", "category": "sexual_violence"},
        
        # Emotional Violence - Swahili
        {"text": "Ananichukiza na kunidharau mbele ya watu", "category": "emotional_violence"},
        {"text": "Ananifanya nihisi sina thamani kabisa", "category": "emotional_violence"},
        {"text": "Ananisema vibaya na kunishutumu kila wakati", "category": "emotional_violence"},
        {"text": "Amenikana uhuru wa kukutana na marafiki zangu", "category": "emotional_violence"},
        {"text": "Ananidhalilisha na kunifanya nijisikie mbaya", "category": "emotional_violence"},
        {"text": "Ananishutumu kwa makosa yake mwenyewe", "category": "emotional_violence"},
        {"text": "Ananicheka na kunibeza daima", "category": "emotional_violence"},
        {"text": "Ananisema niko mjinga na sina akili", "category": "emotional_violence"},
        
        # Emotional Violence - English  
        {"text": "He constantly insults and humiliates me in public", "category": "emotional_violence"},
        {"text": "He makes me feel worthless and stupid every day", "category": "emotional_violence"},
        {"text": "He verbally abuses me and calls me horrible names", "category": "emotional_violence"},
        {"text": "He isolates me from my friends and family", "category": "emotional_violence"},
        {"text": "He belittles everything I do and say", "category": "emotional_violence"},
        {"text": "He constantly criticizes my appearance and intelligence", "category": "emotional_violence"},
        {"text": "He gaslights me and makes me doubt my sanity", "category": "emotional_violence"},
        
        # Economic Violence - Swahili
        {"text": "Ananinyima pesa za chakula na mahitaji ya nyumbani", "category": "economic_violence"},
        {"text": "Hananipi fedha za kujikimu na watoto", "category": "economic_violence"},
        {"text": "Anadhibiti pesa zangu zote na mali yangu", "category": "economic_violence"},
        {"text": "Amenikana nafasi ya kufanya kazi kupata pesa", "category": "economic_violence"},
        {"text": "Anachukua mshahara wangu wote kila mwezi", "category": "economic_violence"},
        {"text": "Ananinyima huduma za matibabu na elimu", "category": "economic_violence"},
        
        # Economic Violence - English
        {"text": "He controls all the money and won't give me any", "category": "economic_violence"},
        {"text": "He takes my salary and refuses to give me access", "category": "economic_violence"},
        {"text": "He prevents me from working or earning money", "category": "economic_violence"},
        {"text": "He denies me money for basic needs and food", "category": "economic_violence"},
        {"text": "He controls all financial decisions without consulting me", "category": "economic_violence"},
        {"text": "He refuses to pay for medical care for me and the children", "category": "economic_violence"},
    ]
    
    return pd.DataFrame(manual_data)

def main():
    """
    Main execution function
    """
    # Inspect original dataset
    df, issues = inspect_dataset('gbv_data/Train.csv')
    
    # Clean the dataset
    df_cleaned = clean_dataset(
        df, 
        remove_duplicates=True, 
        min_length=10
    )
    
    # Add manual Swahili examples
    print("\n" + "="*70)
    print("ADDING MANUAL SWAHILI EXAMPLES")
    print("="*70)
    manual_df = create_manual_swahili_dataset()
    print(f"Added {len(manual_df)} high-quality manual examples")
    
    # Combine datasets
    final_df = pd.concat([df_cleaned, manual_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save cleaned dataset
    final_df.to_csv('gbv_cleaned_dataset.csv', index=False)
    print(f"\n✓ Saved cleaned dataset: gbv_cleaned_dataset.csv")
    print(f"✓ Final size: {len(final_df)} samples")
    print(f"\nFinal distribution:")
    print(final_df['type'].value_counts())
    
    print("\n" + "="*70)
    print("✓ DATASET READY FOR TRAINING!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review potential_mislabeled.csv if it was created")
    print("2. Use 'gbv_cleaned_dataset.csv' for training")
    print("3. Run: python train_model.py")
    
if __name__ == "__main__":
    main()