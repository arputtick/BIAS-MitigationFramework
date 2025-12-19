#!/usr/bin/env python3
"""
Profession-level TPR Gap Analysis from Existing Results

This script analyzes existing evaluation results to extract profession-level
True Positive Rate (TPR) gaps and generates LaTeX tables showing:
- Top 3 professions favoring males (highest positive gaps)
- Top 3 professions favoring females (highest negative gaps) 
- Top 3 professions with smallest bias (gaps closest to zero)

Usage:
    python profession_analysis.py [--results_dir PATH] [--languages LANG1,LANG2,...]

Author: Generated for bias mitigation research
"""

import json
import os
import glob
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any


def load_profession_names(data_path="../data/bios/"):
    """Load profession names from a sample dataset file."""
    import pandas as pd
    
    # Try to find any CSV file to get profession names
    csv_files = glob.glob(os.path.join(data_path, "**/train_split_balanced.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(data_path, "**/train_split.csv"), recursive=True)
    
    if not csv_files:
        print(f"Warning: No dataset files found in {data_path}")
        return None
        
    # Load the first available file
    df = pd.read_csv(csv_files[0])
    professions = sorted(df['profession'].unique())
    print(f"Loaded {len(professions)} profession names from {csv_files[0]}")
    return professions


def extract_tpr_from_confusion_matrix(cm_group0, cm_group1, class_labels):
    """
    Extract TPR (True Positive Rate) for each profession from confusion matrices.
    
    Args:
        cm_group0: Confusion matrix for group 0 (females)
        cm_group1: Confusion matrix for group 1 (males)
        class_labels: List of class labels (profession indices)
        
    Returns:
        Dict with TPR values and gaps for each profession
    """
    cm_group0 = np.array(cm_group0)
    cm_group1 = np.array(cm_group1)
    
    profession_tpr_data = {}
    
    for i, class_label in enumerate(class_labels):
        if i >= len(cm_group0) or i >= len(cm_group1):
            continue
            
        # TPR = True Positives / (True Positives + False Negatives)
        # TPR = diagonal element / sum of row
        
        # Group 0 (Female) TPR
        if i < cm_group0.shape[0]:
            tp_female = cm_group0[i, i]  # True positives (diagonal)
            total_female = np.sum(cm_group0[i, :])  # All actual positives for this class
            tpr_female = tp_female / total_female if total_female > 0 else 0
        else:
            tpr_female = 0
            total_female = 0
            
        # Group 1 (Male) TPR
        if i < cm_group1.shape[0]:
            tp_male = cm_group1[i, i]  # True positives (diagonal)
            total_male = np.sum(cm_group1[i, :])  # All actual positives for this class
            tpr_male = tp_male / total_male if total_male > 0 else 0
        else:
            tpr_male = 0
            total_male = 0
            
        # Calculate gap (Male - Female)
        tpr_gap = tpr_male - tpr_female
        
        profession_tpr_data[class_label] = {
            'female_tpr': tpr_female,
            'male_tpr': tpr_male,
            'tpr_gap': tpr_gap,
            'female_count': int(total_female),
            'male_count': int(total_male)
        }
    
    return profession_tpr_data


def analyze_profession_gaps(results_data, profession_names):
    """
    Analyze profession-level TPR gaps from results data.
    
    Args:
        results_data: Dictionary containing evaluation results
        profession_names: List of profession names
        
    Returns:
        Dictionary with top professions in each category
    """
    if 'Group 0 confusion matrix' not in results_data or 'Group 1 confusion matrix' not in results_data:
        print("Warning: Confusion matrices not found in results data")
        return None
        
    cm_group0 = results_data['Group 0 confusion matrix']
    cm_group1 = results_data['Group 1 confusion matrix']
    class_labels = results_data.get('class_labels', list(range(len(cm_group0))))
    
    # Extract TPR data
    tpr_data = extract_tpr_from_confusion_matrix(cm_group0, cm_group1, class_labels)
    
    # Convert to list format with profession names
    profession_list = []
    for class_idx, data in tpr_data.items():
        if class_idx < len(profession_names):
            profession_list.append({
                'profession': profession_names[class_idx],
                'profession_idx': class_idx,
                **data
            })
    
    if not profession_list:
        print("Warning: No profession data extracted")
        return None
        
    # Filter out professions with insufficient data
    profession_list = [p for p in profession_list if p['female_count'] > 0 and p['male_count'] > 0]
    
    if len(profession_list) < 3:
        print(f"Warning: Only {len(profession_list)} professions with sufficient data")
        return None
        
    # Sort by TPR gap
    profession_list.sort(key=lambda x: x['tpr_gap'])
    
    # Get top/bottom professions
    largest_female_favor = profession_list[:3]  # Most negative gaps (favor females)
    smallest_gaps = sorted(profession_list, key=lambda x: abs(x['tpr_gap']))[:3]
    largest_male_favor = profession_list[-3:][::-1]  # Most positive gaps (favor males)
    
    return {
        'all_professions': profession_list,
        'largest_female_favor': largest_female_favor,
        'smallest_gaps': smallest_gaps,
        'largest_male_favor': largest_male_favor
    }


def create_profession_tpr_gap_table(profession_analysis, language=None):
    """
    Create LaTeX table showing professions with largest TPR gaps.
    
    Args:
        profession_analysis: Dictionary from analyze_profession_gaps function
        language: Language code for table caption
        
    Returns:
        str: LaTeX table code
    """
    if not profession_analysis:
        return "% No profession analysis data available"
        
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    if language:
        lang_display = language.upper()
        latex_lines.append(f"\\caption{{{lang_display} Profession-Level TPR Gap Analysis}}")
        latex_lines.append(f"\\label{{tab:{language}_profession_tpr_gaps}}")
    else:
        latex_lines.append("\\caption{Profession-Level TPR Gap Analysis}")
        latex_lines.append("\\label{tab:profession_tpr_gaps}")
    
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Category & Profession & Female TPR & Male TPR & Gap (M-F) \\\\")
    latex_lines.append("\\midrule")
    
    # Most Female-Favored Professions
    latex_lines.append("\\multirow{3}{*}{\\rotatebox{90}{Female Favored}} ")
    for i, prof in enumerate(profession_analysis['largest_female_favor']):
        prefix = "& " if i > 0 else ""
        prof_name = prof['profession'].replace('_', '\\_')
        latex_lines.append(f"{prefix}{prof_name} & "
                          f"{prof['female_tpr']:.3f} & {prof['male_tpr']:.3f} & "
                          f"{prof['tpr_gap']:.3f} \\\\")
    
    latex_lines.append("\\midrule")
    
    # Smallest Gap Professions
    latex_lines.append("\\multirow{3}{*}{\\rotatebox{90}{Smallest Gap}} ")
    for i, prof in enumerate(profession_analysis['smallest_gaps']):
        prefix = "& " if i > 0 else ""
        prof_name = prof['profession'].replace('_', '\\_')
        latex_lines.append(f"{prefix}{prof_name} & "
                          f"{prof['female_tpr']:.3f} & {prof['male_tpr']:.3f} & "
                          f"{prof['tpr_gap']:.3f} \\\\")
    
    latex_lines.append("\\midrule")
    
    # Most Male-Favored Professions
    latex_lines.append("\\multirow{3}{*}{\\rotatebox{90}{Male Favored}} ")
    for i, prof in enumerate(profession_analysis['largest_male_favor']):
        prefix = "& " if i > 0 else ""
        prof_name = prof['profession'].replace('_', '\\_')
        latex_lines.append(f"{prefix}{prof_name} & "
                          f"{prof['female_tpr']:.3f} & {prof['male_tpr']:.3f} & "
                          f"{prof['tpr_gap']:.3f} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\footnotesize")
    latex_lines.append("\\item Note: Gap = Male TPR - Female TPR. Positive values favor males, negative values favor females.")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)


def find_results_files(results_dir, languages=None):
    """
    Find all results files in the specified directory.
    
    Args:
        results_dir: Path to results directory
        languages: List of language codes to process (None = all)
        
    Returns:
        Dictionary mapping language to list of results files
    """
    results_files = {}
    
    if languages is None:
        # Auto-detect languages from directory structure
        lang_dirs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d)) 
                    and len(d) <= 3]  # Assume language codes are 2-3 chars
        languages = sorted(lang_dirs)
    
    for lang in languages:
        lang_dir = os.path.join(results_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Warning: Directory not found for language '{lang}': {lang_dir}")
            continue
            
        # Find all JSON results files
        pattern = os.path.join(lang_dir, "*_results.json")
        files = glob.glob(pattern)
        
        if files:
            results_files[lang] = files
            print(f"Found {len(files)} results files for {lang}: {[os.path.basename(f) for f in files]}")
        else:
            print(f"Warning: No results files found for language '{lang}' in {lang_dir}")
    
    return results_files


def process_language_results(lang, files_list, profession_names, output_dir):
    """
    Process all results files for a single language.
    
    Args:
        lang: Language code
        files_list: List of results files for this language
        profession_names: List of profession names
        output_dir: Output directory for tables
        
    Returns:
        Number of tables generated
    """
    tables_created = 0
    
    for results_file in files_list:
        try:
            # Load results
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            # Analyze profession gaps
            profession_analysis = analyze_profession_gaps(results_data, profession_names)
            
            if profession_analysis:
                # Create LaTeX table
                latex_table = create_profession_tpr_gap_table(profession_analysis, language=lang)
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(results_file))[0]
                output_name = f"{base_name}_profession_tpr_gaps"
                
                # Save LaTeX file
                tex_file = os.path.join(output_dir, lang, f"{output_name}.tex")
                os.makedirs(os.path.dirname(tex_file), exist_ok=True)
                with open(tex_file, 'w') as f:
                    f.write(latex_table)
                
                # Save text file
                txt_file = os.path.join(output_dir, lang, f"{output_name}.txt")
                with open(txt_file, 'w') as f:
                    f.write(latex_table)
                
                print(f"  ✓ Created tables for {base_name}")
                print(f"    LaTeX: {tex_file}")
                print(f"    Text: {txt_file}")
                
                # Print summary
                print(f"    Female-favored: {', '.join([p['profession'] for p in profession_analysis['largest_female_favor']])}")
                print(f"    Male-favored: {', '.join([p['profession'] for p in profession_analysis['largest_male_favor']])}")
                print(f"    Smallest gaps: {', '.join([p['profession'] for p in profession_analysis['smallest_gaps']])}")
                
                tables_created += 1
            else:
                print(f"  ✗ Could not analyze {os.path.basename(results_file)}")
                
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(results_file)}: {e}")
    
    return tables_created


def main():
    """Main function to process all results and generate profession analysis tables."""
    parser = argparse.ArgumentParser(description='Analyze profession-level TPR gaps from existing results')
    parser.add_argument('--results_dir', type=str, default='../results/adv_debias/',
                       help='Directory containing results files (default: ../results/adv_debias/)')
    parser.add_argument('--languages', type=str, default=None,
                       help='Comma-separated list of languages to process (default: auto-detect)')
    parser.add_argument('--data_path', type=str, default='../data/bios/',
                       help='Path to dataset files for profession names (default: ../data/bios/)')
    parser.add_argument('--output_dir', type=str, default='../results/profession_analysis/',
                       help='Output directory for generated tables (default: ../results/profession_analysis/)')
    
    args = parser.parse_args()
    
    print("🔍 PROFESSION-LEVEL TPR GAP ANALYSIS")
    print("=" * 50)
    
    # Parse languages
    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(',')]
        print(f"Processing specified languages: {languages}")
    else:
        print("Auto-detecting languages from results directory")
    
    # Load profession names
    print(f"\n📋 Loading profession names from {args.data_path}")
    profession_names = load_profession_names(args.data_path)
    if not profession_names:
        print("❌ Could not load profession names. Please check data path.")
        return 1
    
    # Find results files
    print(f"\n📂 Searching for results files in {args.results_dir}")
    results_files = find_results_files(args.results_dir, languages)
    
    if not results_files:
        print("❌ No results files found. Please check results directory.")
        return 1
    
    # Process each language
    total_tables = 0
    print(f"\n📊 Processing {len(results_files)} languages")
    
    for lang, files_list in results_files.items():
        print(f"\n🔤 Processing {lang.upper()} ({len(files_list)} files)")
        tables_created = process_language_results(lang, files_list, profession_names, args.output_dir)
        total_tables += tables_created
    
    # Summary
    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"Generated {total_tables} profession analysis tables")
    print(f"Output directory: {args.output_dir}")
    print("\nEach table shows:")
    print("• Top 3 professions favoring females (most negative gaps)")
    print("• Top 3 professions with smallest bias (gaps closest to zero)")  
    print("• Top 3 professions favoring males (most positive gaps)")
    print("\nGap = Male TPR - Female TPR")
    print("• Positive values = Male advantage")
    print("• Negative values = Female advantage")
    print("• Values near zero = Fair/equal performance")
    
    return 0


if __name__ == "__main__":
    exit(main())
