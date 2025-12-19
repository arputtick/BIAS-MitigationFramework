import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
import glob

def find_available_bias_tests(language, base_data_dir="../data/wordlists/"):
    """
    Find all available WEAT and LPBS tests for a given language.
    """
  # Map language codes if needed
    lang_mapping = {
        'de': 'de', 'en': 'en', 'fr': 'fr', 'is': 'is', 
        'it': 'it', 'nl': 'nl', 'nb': 'nb', 'tk': 'tk'
    }
    
    dataset_lang = lang_mapping.get(language, language)
    
    weat_tests = []
    lpbs_tests = []
    
    # Find WEAT tests
    weat_path = Path(base_data_dir) / dataset_lang / "WEAT"
    if weat_path.exists():
        weat_files = list(weat_path.glob("*.txt"))
        weat_tests = [f.stem for f in weat_files]  # stem removes .txt extension
    
    # Find LPBS tests
    lpbs_path = Path(base_data_dir) / dataset_lang / "LPBS"
    if lpbs_path.exists():
        lpbs_files = list(lpbs_path.glob("*.jsonl"))
        # Extract base test names from LPBS files (remove LPBS_ prefix if present)
        for f in lpbs_files:
            test_name = f.stem
            if test_name.startswith('LPBS_'):
                test_name = test_name[5:]  # Remove 'LPBS_' prefix
            lpbs_tests.append(test_name)
    
    print(f"Language {language}: Found {len(weat_tests)} WEAT tests, {len(lpbs_tests)} LPBS tests")
    print(f"  WEAT: {weat_tests}")
    print(f"  LPBS: {lpbs_tests}")
    
    return weat_tests, lpbs_tests

def extract_experiment_info(filename):
    """
    Extract experiment configuration from filename.
    
    Example: 'bert_adv_debiasing_1discriminators_balanced_masked_results.json'
    Returns: dict with experiment parameters
    """
    # Remove file extension and various suffixes
    name = filename.replace('_leakage_results.json', '').replace('_intrinsic_results.txt', '').replace('_results.json', '')
    
    parts = name.split('_')
    
    config = {
        'model': 'bert',  # Default
        'method': 'standard',  # Default
        'discriminators': 0,
        'masked': False,
        'balanced': False
    }
    
    # Parse filename parts
    if 'adv_debiasing' in name or 'adversarial' in name:
        config['method'] = 'adversarial'
    elif 'standard' in name:
        config['method'] = 'standard'
    elif 'pretrained' in name:
        config['method'] = 'pretrained'
    
    # Extract number of discriminators
    for part in parts:
        if 'discriminator' in part:
            try:
                config['discriminators'] = int(part.replace('discriminators', '').replace('discriminator', ''))
            except:
                config['discriminators'] = 1
    
    # Check for masking and balancing - be more explicit
    config['masked'] = 'masked' in name
    config['balanced'] = 'balanced' in name
    
    # Debug print to see what's being parsed
    print(f"  Parsing filename: {filename}")
    print(f"  -> method: {config['method']}, masked: {config['masked']}, balanced: {config['balanced']}")
    
    return config

def load_fairness_results(results_dir):
    """
    Load all fairness results (main task prediction bias) from the specified directory.
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist")
        return pd.DataFrame()
    
    all_results = []
    
    # Find all main results files (not leakage or intrinsic)
    results_files = [f for f in results_dir.glob('*_results.json') 
                    if '_leakage_results' not in f.name and '_intrinsic_results' not in f.name]
    
    if not results_files:
        print(f"No fairness results files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(results_files)} fairness results files:")
    
    for file_path in results_files:
        print(f"  - {file_path.name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract experiment configuration from filename
            config = extract_experiment_info(file_path.name)
            
            # Create result row
            result = {
                'filename': file_path.name,
                'method': config['method'],
                'discriminators': config['discriminators'],
                'masked': config['masked'],
                'balanced': config['balanced'],
                # Main fairness metrics
                'accuracy_gap': data.get('Accuracy_gap', np.nan),
                'f1_gap': data.get('F1_gap', np.nan),
                'eq_odds_gap': data.get('eq_odds_gap', np.nan),
                # Individual group performance
                'accuracy_0': data.get('Accuracy_0', np.nan),
                'accuracy_1': data.get('Accuracy_1', np.nan),
                'f1_macro_0': data.get('F1_macro_0', np.nan),
                'f1_macro_1': data.get('F1_macro_1', np.nan),
                # Equalized odds breakdown
                'max_tpr_gap': data.get('max_tpr_gap', np.nan),
                'max_fpr_gap': data.get('max_fpr_gap', np.nan),
                'avg_tpr_gap': data.get('avg_tpr_gap', np.nan),
                'avg_fpr_gap': data.get('avg_fpr_gap', np.nan),
                # Dataset info
                'num_classes': data.get('num_classes', np.nan),
                'class_labels': data.get('class_labels', [])
            }
            
            # Legacy binary classification metrics (if available)
            if 'TPR_gap' in data:
                result.update({
                    'tpr_gap': data.get('TPR_gap', np.nan),
                    'tnr_gap': data.get('TNR_gap', np.nan),
                    'tpr_0': data.get('TPR_0', np.nan),
                    'tpr_1': data.get('TPR_1', np.nan),
                    'tnr_0': data.get('TNR_0', np.nan),
                    'tnr_1': data.get('TNR_1', np.nan)
                })
            
            # Store confusion matrices if available
            if 'Group 0 confusion matrix' in data:
                result['confusion_matrix_0'] = data['Group 0 confusion matrix']
                result['confusion_matrix_1'] = data['Group 1 confusion matrix']
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    if not all_results:
        print("No valid fairness results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    return df

def parse_intrinsic_results_dynamic(content, language="en"):
    """
    Parse intrinsic bias results for all available WEAT and LPBS tests dynamically.
    """
    # Find all available tests for this language
    weat_tests, lpbs_tests = find_available_bias_tests(language)
    
    result = {}
    lines = content.strip().split('\n')
    
    for line in lines:
        try:
            # Parse WEAT tests - look for test name directly (no WEAT_ prefix)
            for test_name in weat_tests:
                if test_name in line and 'LPBS' not in line:
                    effect_match = re.search(r"'Effect Size':\s*np\.float(?:32|64)?\(([0-9.-]+)\)", line)
                    pval_match = re.search(r"'p-value':\s*([0-9.-]+)", line)
                    
                    if effect_match and pval_match:
                        result[f'weat_{test_name}_effect_size'] = float(effect_match.group(1))
                        result[f'weat_{test_name}_p_value'] = float(pval_match.group(1))
            
            # Parse LPBS tests - handle the LPBS_{test_name} format
            for test_name in lpbs_tests:
                # LPBS tests appear as "LPBS_{test_name}" in results
                if f"LPBS_{test_name}" in line or (test_name in line and 'LPBS' in line):
                    effect_match = re.search(r"'Effect Size':\s*np\.float(?:32|64)?\(([0-9.-]+)\)", line)
                    pval_match = re.search(r"'p-value':\s*([0-9.-]+)", line)
                    
                    if effect_match and pval_match:
                        # Store with the original test name (removing LPBS_ prefix if it exists in test_name)
                        base_test_name = test_name.replace('LPBS_', '') if test_name.startswith('LPBS_') else test_name
                        result[f'lpbs_LPBS_{base_test_name}_effect_size'] = float(effect_match.group(1))
                        result[f'lpbs_LPBS_{base_test_name}_p_value'] = float(pval_match.group(1))
                        
        except Exception as line_error:
            continue
    
    return result

def load_leakage_results(results_dir):
    """
    Load all leakage results from the specified directory.
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist")
        return pd.DataFrame()
    
    all_results = []
    
    # Find all leakage results files
    leakage_files = list(results_dir.glob('*leakage_results*.json'))
    
    if not leakage_files:
        print(f"No leakage results files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(leakage_files)} leakage results files:")
    
    for file_path in leakage_files:
        print(f"  - {file_path.name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract experiment configuration from filename
            config = extract_experiment_info(file_path.name)
            
            # Create result row
            result = {
                'filename': file_path.name,
                'method': config['method'],
                'discriminators': config['discriminators'],
                'masked': config['masked'],
                'balanced': config['balanced'],
                'train_leakage': data.get('Train Leakage', data.get('train_leakage', np.nan)),
                'test_leakage': data.get('Test Leakage', data.get('test_leakage', np.nan)),
                'test_f1': data.get('Test F1 Score', data.get('test_f1', np.nan)),
                'test_roc_auc': data.get('Test ROC AUC', data.get('test_roc_auc', np.nan)),
                'test_avg_precision': data.get('Test Average Precision', data.get('test_avg_precision', np.nan))
            }
            
            # Extract confusion matrix if available
            if 'Confusion Matrix' in data:
                cm = np.array(data['Confusion Matrix'])
                result['confusion_matrix'] = cm.tolist()
                
                # Calculate additional metrics from confusion matrix
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    result['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                    result['precision'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    result['recall'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    if not all_results:
        print("No valid results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    return df

def load_intrinsic_results(results_dir, language="en"):
    """
    Load all intrinsic bias results from the specified directory with dynamic test discovery.
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist")
        return pd.DataFrame()
    
    all_results = []
    
    # Find all intrinsic results files
    intrinsic_files = list(results_dir.glob('*intrinsic_results*.txt'))
    
    if not intrinsic_files:
        print(f"No intrinsic results files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(intrinsic_files)} intrinsic results files:")
    
    for file_path in intrinsic_files:
        print(f"  - {file_path.name}")
        
        try:
            # Extract experiment configuration from filename
            config = extract_experiment_info(file_path.name)
            
            # Parse the intrinsic results file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Initialize result row with basic info
            result = {
                'filename': file_path.name,
                'method': config['method'],
                'discriminators': config['discriminators'],
                'masked': config['masked'],
                'balanced': config['balanced']
            }
            
            # Parse all available test results dynamically
            parsed_results = parse_intrinsic_results_dynamic(content, language)
            result.update(parsed_results)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    if not all_results:
        print("No valid intrinsic results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    return df

def create_fairness_latex_table(df, multilingual=False, language=None):
    """
    Create LaTeX table for main task fairness results (multiclass).
    Assumes all datasets are balanced, so removes balanced column.
    
    Args:
        df: DataFrame with fairness results
        multilingual: If True, includes language column for cross-language comparison
        language: Language code for single-language tables (to include in caption)
    """
    if df.empty:
        return "No data available for fairness table generation."
    
    # Determine sort columns and merge columns based on multilingual flag
    if multilingual:
        sort_cols = ['language', 'method', 'discriminators', 'masked']
        base_cols = 4  # language, method, discriminators, masked
    else:
        sort_cols = ['method', 'discriminators', 'masked']
        base_cols = 3  # method, discriminators, masked
    
    # Sort dataframe
    df_sorted = df.sort_values(sort_cols)
    
    latex_lines = []
    
    # Table header - includes language column if multilingual
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    if multilingual:
        latex_lines.append("\\caption{Multilingual Main Task Fairness Results}")
        latex_lines.append("\\label{tab:multilingual_fairness_results}")
        latex_lines.append("\\begin{tabular}{llcccccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Lang & Method & Disc. & Masked & Acc Gap & F1 Gap & EqOdds Gap & Acc G0 & Acc G1 \\\\")
    else:
        # Include language in caption for single-language tables
        if language:
            lang_display = language.upper()
            latex_lines.append(f"\\caption{{{lang_display} Main Task Fairness Results}}")
            latex_lines.append(f"\\label{{tab:{language}_fairness_results}}")
        else:
            latex_lines.append("\\caption{Main Task Fairness Results}")
            latex_lines.append("\\label{tab:fairness_results}")
        latex_lines.append("\\begin{tabular}{lccccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Method & Disc. & Masked & Acc Gap & F1 Gap & EqOdds Gap & Acc G0 & Acc G1 \\\\")
    
    latex_lines.append("\\midrule")
    
    # Table rows
    for _, row in df_sorted.iterrows():
        method = row['method'].replace('_', '\\_')
        discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
        masked = 'Yes' if row['masked'] else 'No'
        
        # Format numbers to 3 decimal places
        acc_gap = f"{row['accuracy_gap']:.3f}" if not pd.isna(row['accuracy_gap']) else '-'
        f1_gap = f"{row['f1_gap']:.3f}" if not pd.isna(row['f1_gap']) else '-'
        avg_tpr_gap = f"{row['avg_tpr_gap']:.3f}" if not pd.isna(row['avg_tpr_gap']) else '-'
        acc_0 = f"{row['accuracy_0']:.3f}" if not pd.isna(row['accuracy_0']) else '-'
        acc_1 = f"{row['accuracy_1']:.3f}" if not pd.isna(row['accuracy_1']) else '-'
        
        if multilingual:
            language = row['language'].upper()
            latex_lines.append(f"{language} & {method} & {discriminators} & {masked} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {acc_0} & {acc_1} \\\\")
        else:
            latex_lines.append(f"{method} & {discriminators} & {masked} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {acc_0} & {acc_1} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_equalized_odds_detailed_table(df, language=None):
    """
    Create detailed LaTeX table for equalized odds breakdown.
    
    Args:
        df: DataFrame with fairness results
        language: Language code for single-language tables (to include in caption)
    """
    if df.empty:
        return "No data available for equalized odds detailed table generation."
    
    # Sort by method, then by discriminators, then by masked
    df_sorted = df.sort_values(['method', 'discriminators', 'masked'])
    
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    # Include language in caption for single-language tables
    if language:
        lang_display = language.upper()
        latex_lines.append(f"\\caption{{{lang_display} Equalized Odds Detailed Breakdown}}")
        latex_lines.append(f"\\label{{tab:{language}_eq_odds_detailed}}")
    else:
        latex_lines.append("\\caption{Equalized Odds Detailed Breakdown}")
        latex_lines.append("\\label{tab:eq_odds_detailed}")
        
    latex_lines.append("\\begin{tabular}{lcccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Method & Disc. & Masked & EqOdds Gap & Max TPR Gap & Max FPR Gap & Avg TPR Gap & Avg FPR Gap \\\\")
    latex_lines.append("\\midrule")
    
    # Table rows
    for _, row in df_sorted.iterrows():
        method = row['method'].replace('_', '\\_')
        discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
        masked = 'Yes' if row['masked'] else 'No'
        
        # Format numbers to 3 decimal places
        eq_odds_gap = f"{row['eq_odds_gap']:.3f}" if not pd.isna(row['eq_odds_gap']) else '-'
        max_tpr_gap = f"{row['max_tpr_gap']:.3f}" if not pd.isna(row['max_tpr_gap']) else '-'
        max_fpr_gap = f"{row['max_fpr_gap']:.3f}" if not pd.isna(row['max_fpr_gap']) else '-'
        avg_tpr_gap = f"{row['avg_tpr_gap']:.3f}" if not pd.isna(row['avg_tpr_gap']) else '-'
        avg_fpr_gap = f"{row['avg_fpr_gap']:.3f}" if not pd.isna(row['avg_fpr_gap']) else '-'
        
        latex_lines.append(f"{method} & {discriminators} & {masked} & {eq_odds_gap} & {max_tpr_gap} & {max_fpr_gap} & {avg_tpr_gap} & {avg_fpr_gap} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_intrinsic_latex_table(df, multilingual=False, language=None, show_details=False):
    """
    Create LaTeX table for intrinsic bias results showing WEAT/LPBS test pairs.
    Uses a stacked format with a Test column instead of multicolumn headers.
    
    Args:
        df: DataFrame with intrinsic results
        multilingual: If True, includes language column for cross-language comparison
        language: Language code for single-language tables (to include in caption)
        show_details: If True, includes discriminator and masked columns
    """
    if df.empty:
        return "No data available for intrinsic bias table generation."
    
    # Find all effect size columns dynamically
    effect_size_columns = [col for col in df.columns if col.endswith('_effect_size')]
    
    # Extract test pairs - group WEAT and LPBS tests by base test name
    weat_tests = set()
    lpbs_tests = set()
    
    for col in effect_size_columns:
        if col.startswith('weat_'):
            test_name = col.replace('weat_', '').replace('_effect_size', '')
            weat_tests.add(test_name)
        elif col.startswith('lpbs_'):
            # LPBS tests are in format: lpbs_LPBS_{test_name}_effect_size
            test_name = col.replace('lpbs_LPBS_', '').replace('lpbs_', '').replace('_effect_size', '')
            # Handle case where LPBS test name includes the base test name
            if test_name.startswith('LPBS_'):
                test_name = test_name.replace('LPBS_', '')
            lpbs_tests.add(test_name)
    
    # Find test pairs - tests that exist in both WEAT and LPBS
    test_pairs = []
    all_test_names = weat_tests.union(lpbs_tests)
    
    for test_name in sorted(all_test_names):
        weat_effect_col = f'weat_{test_name}_effect_size'
        weat_pval_col = f'weat_{test_name}_p_value'
        
        # Try different LPBS column name formats
        lpbs_effect_col = None
        lpbs_pval_col = None
        
        possible_lpbs_formats = [
            f'lpbs_LPBS_{test_name}_effect_size',
            f'lpbs_{test_name}_effect_size',
            f'lpbs_LPBS{test_name}_effect_size'  # In case there's no underscore
        ]
        
        for lpbs_format in possible_lpbs_formats:
            if lpbs_format in df.columns:
                lpbs_effect_col = lpbs_format
                lpbs_pval_col = lpbs_format.replace('_effect_size', '_p_value')
                break
        
        # Check if we have data for this test
        has_weat = weat_effect_col in df.columns and df[weat_effect_col].notna().any()
        has_lpbs = lpbs_effect_col and lpbs_effect_col in df.columns and df[lpbs_effect_col].notna().any()
        
        if has_weat or has_lpbs:
            test_pairs.append({
                'test_name': test_name,
                'weat_effect_col': weat_effect_col if has_weat else None,
                'weat_pval_col': weat_pval_col if has_weat else None,
                'lpbs_effect_col': lpbs_effect_col if has_lpbs else None,
                'lpbs_pval_col': lpbs_pval_col if has_lpbs else None
            })
    
    if not test_pairs:
        return "No intrinsic bias test pairs found."
    
    # Sort test pairs by test name for consistent ordering
    test_pairs = sorted(test_pairs, key=lambda x: x['test_name'])
    
    # Create stacked data
    stacked_rows = []
    
    # Determine sort columns based on multilingual flag and show_details - don't include test name in initial sort
    if multilingual:
        if show_details:
            base_sort_cols = ['language', 'method', 'discriminators', 'masked']
        else:
            base_sort_cols = ['language', 'method']
    else:
        if show_details:
            base_sort_cols = ['method', 'discriminators', 'masked']
        else:
            base_sort_cols = ['method']
    
    # Sort dataframe by experimental conditions first
    df_sorted = df.sort_values(base_sort_cols)
    
    # Create rows for each test and each experiment
    for _, row in df_sorted.iterrows():
        for test_pair in test_pairs:  # test_pairs is already sorted by test name
            # Extract values for this test pair
            weat_effect = '-'
            weat_pval = '-'
            if test_pair['weat_effect_col'] and test_pair['weat_effect_col'] in row:
                if not pd.isna(row[test_pair['weat_effect_col']]):
                    weat_effect = f"{row[test_pair['weat_effect_col']]:.3f}"
            if test_pair['weat_pval_col'] and test_pair['weat_pval_col'] in row:
                if not pd.isna(row[test_pair['weat_pval_col']]):
                    weat_pval = f"{row[test_pair['weat_pval_col']]:.3f}"
            
            lpbs_effect = '-'
            lpbs_pval = '-'
            if test_pair['lpbs_effect_col'] and test_pair['lpbs_effect_col'] in row:
                if not pd.isna(row[test_pair['lpbs_effect_col']]):
                    lpbs_effect = f"{row[test_pair['lpbs_effect_col']]:.3f}"
            if test_pair['lpbs_pval_col'] and test_pair['lpbs_pval_col'] in row:
                if not pd.isna(row[test_pair['lpbs_pval_col']]):
                    lpbs_pval = f"{row[test_pair['lpbs_pval_col']]:.3f}"
            
            # Only add row if we have at least some data for this test
            if weat_effect != '-' or weat_pval != '-' or lpbs_effect != '-' or lpbs_pval != '-':
                stacked_row = {
                    'test_name': test_pair['test_name'].replace('_', ' ').title(),
                    'test_name_sort': test_pair['test_name'],  # Keep original for sorting
                    'method': row['method'].replace('_', '\\_'),
                    'weat_effect': weat_effect,
                    'weat_pval': weat_pval,
                    'lpbs_effect': lpbs_effect,
                    'lpbs_pval': lpbs_pval
                }
                
                if show_details:
                    stacked_row['discriminators'] = row['discriminators'] if row['discriminators'] > 0 else '-'
                    stacked_row['masked'] = 'Y' if row['masked'] else 'N'
                
                if multilingual:
                    stacked_row['language'] = row['language'].upper()
                
                stacked_rows.append(stacked_row)
    
    if not stacked_rows:
        return "No intrinsic bias data found for table generation."
    
    # Sort the final stacked rows by test name first, then by experimental conditions
    if multilingual:
        if show_details:
            stacked_rows.sort(key=lambda x: (x['test_name_sort'], x['language'], x['method'], x['discriminators'], x['masked']))
        else:
            stacked_rows.sort(key=lambda x: (x['test_name_sort'], x['language'], x['method']))
    else:
        if show_details:
            stacked_rows.sort(key=lambda x: (x['test_name_sort'], x['method'], x['discriminators'], x['masked']))
        else:
            stacked_rows.sort(key=lambda x: (x['test_name_sort'], x['method']))
    
    latex_lines = []
    
    # Table header - stacked format with Test column
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    if multilingual:
        latex_lines.append("\\caption{Multilingual Intrinsic Bias Results - WEAT/LPBS Test Pairs}")
        latex_lines.append("\\label{tab:multilingual_intrinsic_results}")
        if show_details:
            latex_lines.append("\\begin{tabular}{lllcccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Lang & Test & Method & Disc. & Masked & WEAT ES & WEAT p & LPBS ES & LPBS p \\\\")
        else:
            latex_lines.append("\\begin{tabular}{lllcccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Lang & Test & Method & WEAT ES & WEAT p & LPBS ES & LPBS p \\\\")
    else:
        # Include language in caption for single-language tables
        if language:
            lang_display = language.upper()
            latex_lines.append(f"\\caption{{{lang_display} Intrinsic Bias Results - WEAT/LPBS Test Pairs}}")
            latex_lines.append(f"\\label{{tab:{language}_intrinsic_results}}")
        else:
            latex_lines.append("\\caption{Intrinsic Bias Results - WEAT/LPBS Test Pairs}")
            latex_lines.append("\\label{tab:intrinsic_results}")
        if show_details:
            latex_lines.append("\\begin{tabular}{llcccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Test & Method & Disc. & Masked & WEAT ES & WEAT p & LPBS ES & LPBS p \\\\")
        else:
            latex_lines.append("\\begin{tabular}{llcccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Test & Method & WEAT ES & WEAT p & LPBS ES & LPBS p \\\\")
    
    latex_lines.append("\\midrule")
    
    # Table rows - stacked format (already sorted by test name)
    for stacked_row in stacked_rows:
        if multilingual:
            if show_details:
                row_data = f"{stacked_row['language']} & {stacked_row['test_name']} & {stacked_row['method']} & {stacked_row['discriminators']} & {stacked_row['masked']} & {stacked_row['weat_effect']} & {stacked_row['weat_pval']} & {stacked_row['lpbs_effect']} & {stacked_row['lpbs_pval']} \\\\"
            else:
                row_data = f"{stacked_row['language']} & {stacked_row['test_name']} & {stacked_row['method']} & {stacked_row['weat_effect']} & {stacked_row['weat_pval']} & {stacked_row['lpbs_effect']} & {stacked_row['lpbs_pval']} \\\\"
        else:
            if show_details:
                row_data = f"{stacked_row['test_name']} & {stacked_row['method']} & {stacked_row['discriminators']} & {stacked_row['masked']} & {stacked_row['weat_effect']} & {stacked_row['weat_pval']} & {stacked_row['lpbs_effect']} & {stacked_row['lpbs_pval']} \\\\"
            else:
                row_data = f"{stacked_row['test_name']} & {stacked_row['method']} & {stacked_row['weat_effect']} & {stacked_row['weat_pval']} & {stacked_row['lpbs_effect']} & {stacked_row['lpbs_pval']} \\\\"
        
        latex_lines.append(row_data)
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_comprehensive_latex_table(fairness_df, leakage_df, intrinsic_df, multilingual=False, language=None, show_details=False):
    """
    Create comprehensive LaTeX table with fairness and leakage metrics only (no intrinsic bias).
    
    Args:
        fairness_df: DataFrame with fairness results
        leakage_df: DataFrame with leakage results  
        intrinsic_df: DataFrame with intrinsic results (not used in this table)
        multilingual: If True, includes language column for cross-language comparison
        language: Language code for single-language tables (to include in caption)
        show_details: If True, includes discriminator and masked columns
    """
    if fairness_df.empty and leakage_df.empty:
        return "No data available for comprehensive table generation."
    
    # Determine merge columns based on multilingual flag and show_details
    if multilingual:
        if show_details:
            merge_cols = ['language', 'method', 'discriminators', 'masked', 'balanced']
        else:
            merge_cols = ['language', 'method', 'balanced']
    else:
        if show_details:
            merge_cols = ['method', 'discriminators', 'masked']
        else:
            merge_cols = ['method']
    
    # Start with fairness results as base
    if not fairness_df.empty:
        combined_df = fairness_df[merge_cols + ['accuracy_gap', 'f1_gap', 'avg_tpr_gap', 'accuracy_0', 'accuracy_1']].copy()
    else:
        combined_df = pd.DataFrame()
    
    # Add leakage results
    if not leakage_df.empty:
        leakage_subset = leakage_df[merge_cols + ['test_leakage', 'test_f1']].copy()
        if not combined_df.empty:
            combined_df = pd.merge(combined_df, leakage_subset, on=merge_cols, how='outer')
        else:
            combined_df = leakage_subset
    
    if combined_df.empty:
        return "No data available for comprehensive table."
    
    # Determine sort columns based on multilingual flag and show_details
    if multilingual:
        if show_details:
            sort_cols = ['language', 'method', 'discriminators', 'masked', 'balanced']
        else:
            sort_cols = ['language', 'method', 'balanced']
    else:
        if show_details:
            sort_cols = ['method', 'discriminators', 'masked']
        else:
            sort_cols = ['method']
    
    # Sort dataframe
    df_sorted = combined_df.sort_values(sort_cols)
    
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    if multilingual:
        latex_lines.append("\\caption{Multilingual Comprehensive Bias Results}")
        latex_lines.append("\\label{tab:multilingual_comprehensive}")
        if show_details:
            latex_lines.append("\\begin{tabular}{lcccccccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Lang & Method & D & M & Acc F & Acc M & Acc Gap & F1 Gap & Avg TPR Gap & Test Leak \\\\")
        else:
            latex_lines.append("\\begin{tabular}{lccccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Lang & Method & Acc F & Acc M & Acc Gap & F1 Gap & Avg TPR Gap & Test Leak \\\\")
    else:
        # Include language in caption for single-language tables
        if language:
            lang_display = language.upper()
            latex_lines.append(f"\\caption{{{lang_display} Comprehensive Bias Results}}")
            latex_lines.append(f"\\label{{tab:{language}_comprehensive}}")
        else:
            latex_lines.append("\\caption{Comprehensive Bias Results - Fairness and Leakage}")
            latex_lines.append("\\label{tab:comprehensive_results}")
        if show_details:
            latex_lines.append("\\begin{tabular}{lcccccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Method & D & M & Acc F & Acc M & Acc Gap & F1 Gap & Avg TPR Gap & Test Leak \\\\")
        else:
            latex_lines.append("\\begin{tabular}{lcccccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Method & Acc F & Acc M & Acc Gap & F1 Gap & Avg TPR Gap & Test Leak \\\\")
    
    latex_lines.append("\\midrule")
    
    # Table rows
    for _, row in df_sorted.iterrows():
        method = row['method'].replace('_', '\\_')
        
        # Format numbers to 3 decimal places
        acc_gap = f"{row['accuracy_gap']:.3f}" if not pd.isna(row.get('accuracy_gap', np.nan)) else '-'
        f1_gap = f"{row['f1_gap']:.3f}" if not pd.isna(row.get('f1_gap', np.nan)) else '-'
        avg_tpr_gap = f"{row['avg_tpr_gap']:.3f}" if not pd.isna(row.get('avg_tpr_gap', np.nan)) else '-'
        test_leakage = f"{row['test_leakage']:.3f}" if not pd.isna(row.get('test_leakage', np.nan)) else '-'
        acc_0 = f"{row['accuracy_0']:.3f}" if not pd.isna(row.get('accuracy_0', np.nan)) else '-'
        acc_1 = f"{row['accuracy_1']:.3f}" if not pd.isna(row.get('accuracy_1', np.nan)) else '-'
        
        if multilingual:
            language = row['language'].upper()
            if show_details:
                discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
                masked = 'Y' if row['masked'] else 'N'
                latex_lines.append(f"{language} & {method} & {discriminators} & {masked} & {acc_0} & {acc_1} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {test_leakage} \\\\")
            else:
                latex_lines.append(f"{language} & {method} & {acc_0} & {acc_1} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {test_leakage} \\\\")
        else:
            if show_details:
                discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
                masked = 'Y' if row['masked'] else 'N'
                latex_lines.append(f"{method} & {discriminators} & {masked} & {acc_0} & {acc_1} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {test_leakage} \\\\")
            else:
                latex_lines.append(f"{method} & {acc_0} & {acc_1} & {acc_gap} & {f1_gap} & {avg_tpr_gap} & {test_leakage} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_leakage_latex_table(df, multilingual=False, language=None):
    """
    Create LaTeX table for leakage results.
    Removes balanced column since all datasets are balanced.
    
    Args:
        df: DataFrame with leakage results
        multilingual: If True, includes language column for cross-language comparison
        language: Language code for single-language tables (to include in caption)
    """
    if df.empty:
        return "No data available for leakage table generation."
    
    # Determine sort columns based on multilingual flag
    if multilingual:
        sort_cols = ['language', 'method', 'discriminators', 'masked']
    else:
        sort_cols = ['method', 'discriminators', 'masked']
    
    # Sort dataframe
    df_sorted = df.sort_values(sort_cols)
    
    latex_lines = []
    
    # Table header - includes language column if multilingual
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    if multilingual:
        latex_lines.append("\\caption{Multilingual Gender Leakage Results}")
        latex_lines.append("\\label{tab:multilingual_leakage_results}")
        latex_lines.append("\\begin{tabular}{llcccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Lang & Method & Disc. & Masked & Test Leakage & Test F1 & Test ROC AUC \\\\")
    else:
        # Include language in caption for single-language tables
        if language:
            lang_display = language.upper()
            latex_lines.append(f"\\caption{{{lang_display} Gender Leakage Results}}")
            latex_lines.append(f"\\label{{tab:{language}_leakage_results}}")
        else:
            latex_lines.append("\\caption{Gender Leakage Results}")
            latex_lines.append("\\label{tab:leakage_results}")
        latex_lines.append("\\begin{tabular}{lcccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Method & Disc. & Masked & Test Leakage & Test F1 & Test ROC AUC \\\\")
    
    latex_lines.append("\\midrule")
    
    # Table rows
    for _, row in df_sorted.iterrows():
        method = row['method'].replace('_', '\\_')
        discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
        masked = 'Yes' if row['masked'] else 'No'
        
        # Format numbers to 3 decimal places
        test_leakage = f"{row['test_leakage']:.3f}" if not pd.isna(row['test_leakage']) else '-'
        test_f1 = f"{row['test_f1']:.3f}" if not pd.isna(row['test_f1']) else '-'
        test_roc_auc = f"{row['test_roc_auc']:.3f}" if not pd.isna(row['test_roc_auc']) else '-'
        
        if multilingual:
            language = row['language'].upper()
            latex_lines.append(f"{language} & {method} & {discriminators} & {masked} & {test_leakage} & {test_f1} & {test_roc_auc} \\\\")
        else:
            latex_lines.append(f"{method} & {discriminators} & {masked} & {test_leakage} & {test_f1} & {test_roc_auc} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_summary_latex_table(df):
    """
    Create summary LaTeX table with key metrics only.
    Removes balanced column since all datasets are balanced.
    """
    if df.empty:
        return "No data available for summary table generation."
    
    # Group by main experimental conditions
    summary_data = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        if method == 'adversarial':
            # Group by number of discriminators and masking
            for disc in sorted(method_data['discriminators'].unique()):
                for masked in [False, True]:
                    subset = method_data[(method_data['discriminators'] == disc) & 
                                       (method_data['masked'] == masked)]
                    if not subset.empty:
                        row = subset.iloc[0]  # Take first matching row
                        summary_data.append({
                            'condition': f"Adversarial ({disc}D, {'Masked' if masked else 'Unmasked'})",
                            'test_leakage': row['test_leakage'],
                            'test_f1': row['test_f1'],
                            'test_roc_auc': row['test_roc_auc']
                        })
        else:
            # For standard/pretrained, group by masking
            for masked in [False, True]:
                subset = method_data[method_data['masked'] == masked]
                if not subset.empty:
                    row = subset.iloc[0]
                    summary_data.append({
                        'condition': f"{method.title()} ({'Masked' if masked else 'Unmasked'})",
                        'test_leakage': row['test_leakage'],
                        'test_f1': row['test_f1'],
                        'test_roc_auc': row['test_roc_auc']
                    })
    
    if not summary_data:
        return "No data available for summary table."
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Gender Leakage Results Summary}")
    latex_lines.append("\\label{tab:leakage_summary}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Condition & Test Leakage & Test F1 & Test ROC AUC \\\\")
    latex_lines.append("\\midrule")
    
    for item in summary_data:
        condition = item['condition'].replace('_', '\\_')
        test_leakage = f"{item['test_leakage']:.3f}" if not pd.isna(item['test_leakage']) else '-'
        test_f1 = f"{item['test_f1']:.3f}" if not pd.isna(item['test_f1']) else '-'
        test_roc_auc = f"{item['test_roc_auc']:.3f}" if not pd.isna(item['test_roc_auc']) else '-'
        
        latex_lines.append(f"{condition} & {test_leakage} & {test_f1} & {test_roc_auc} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def create_profession_tpr_gap_table(profession_analysis, language=None):
    """
    Create LaTeX table showing professions with largest TPR gaps.
    
    Args:
        profession_analysis: Dictionary from analyze_profession_tpr_gaps function
        language: Language code for table caption
        
    Returns:
        str: LaTeX table code
    """
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
        latex_lines.append(f"{prefix}{prof['profession'].replace('_', '\\_')} & "
                          f"{prof['female_tpr']:.3f} & {prof['male_tpr']:.3f} & "
                          f"{prof['tpr_gap']:.3f} \\\\")
    
    latex_lines.append("\\midrule")
    
    # Smallest Gap Professions
    latex_lines.append("\\multirow{3}{*}{\\rotatebox{90}{Smallest Gap}} ")
    for i, prof in enumerate(profession_analysis['smallest_gaps']):
        prefix = "& " if i > 0 else ""
        latex_lines.append(f"{prefix}{prof['profession'].replace('_', '\\_')} & "
                          f"{prof['female_tpr']:.3f} & {prof['male_tpr']:.3f} & "
                          f"{prof['tpr_gap']:.3f} \\\\")
    
    latex_lines.append("\\midrule")
    
    # Most Male-Favored Professions
    latex_lines.append("\\multirow{3}{*}{\\rotatebox{90}{Male Favored}} ")
    for i, prof in enumerate(profession_analysis['largest_male_favor']):
        prefix = "& " if i > 0 else ""
        latex_lines.append(f"{prefix}{prof['profession'].replace('_', '\\_')} & "
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

def save_latex_tables(tables_dict, output_dir="../results/", language=None):
    """
    Save LaTeX tables to text files in language-specific directories.
    
    Args:
        tables_dict (dict): Dictionary of table_name: latex_content pairs
        output_dir (str): Base directory to save files (default: ../results/)
        language (str): Language code for saving in language-specific folder
    """
    if language:
        # Save in language-specific directory
        output_path = Path(output_dir) / "adv_debias" / language
    else:
        # Save in general results directory for multilingual tables
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for table_name, table_content in tables_dict.items():
        # Save as .tex file
        tex_file = output_path / f"{table_name}.tex"
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(table_content)
        
        # Save as .txt file for easy viewing
        txt_file = output_path / f"{table_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(table_content)
        
        saved_files[f'{table_name}_tex'] = str(tex_file)
        saved_files[f'{table_name}_txt'] = str(txt_file)
    
    return saved_files

def find_available_languages(base_results_dir="../results/adv_debias/"):
    """
    Find all languages that have results directories.
    """
    base_path = Path(base_results_dir)
    if not base_path.exists():
        print(f"Base results directory {base_path} does not exist")
        return []
    
    # Find all language directories (2-letter codes)
    language_dirs = [d.name for d in base_path.iterdir() 
                    if d.is_dir() and len(d.name) <= 3]  # Allow up to 3 chars for codes like 'nb'
    
    print(f"Found language directories: {language_dirs}")
    return sorted(language_dirs)

def load_results_for_language(results_dir, language):
    """
    Load all types of results for a specific language.
    Returns tuple of (fairness_df, leakage_df, intrinsic_df).
    """
    print(f"Processing language: {language}")
    print(f"  Directory: {results_dir}")
    
    # Check if directory exists
    if not Path(results_dir).exists():
        print(f"  ⚠️  Directory does not exist, skipping...")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Load results
    fairness_df = load_fairness_results(results_dir)
    leakage_df = load_leakage_results(results_dir)
    intrinsic_df = load_intrinsic_results(results_dir, language)

    # Add language column to each dataframe
    if not fairness_df.empty:
        fairness_df['language'] = language
        print(f"  ✓ {len(fairness_df)} fairness results")
    
    if not leakage_df.empty:
        leakage_df['language'] = language
        print(f"  ✓ {len(leakage_df)} leakage results")
    
    if not intrinsic_df.empty:
        intrinsic_df['language'] = language
        print(f"  ✓ {len(intrinsic_df)} intrinsic results")
        # Show discovered tests
        effect_cols = [col for col in intrinsic_df.columns if col.endswith('_effect_size')]
        if effect_cols:
            test_names = [col.replace('_effect_size', '') for col in effect_cols]
            print(f"  📊 Tests: {', '.join(test_names[:3])}{'...' if len(test_names) > 3 else ''}")
    
    if fairness_df.empty and leakage_df.empty and intrinsic_df.empty:
        print(f"  ⚠️  No results found for {language}")
    
    return fairness_df, leakage_df, intrinsic_df

def create_multilingual_summary_table(all_fairness_dfs, all_leakage_dfs, all_intrinsic_dfs):
    """
    Create a summary table showing results across all languages.
    """
    if not any([all_fairness_dfs, all_leakage_dfs, all_intrinsic_dfs]):
        return "No multilingual data available."
    
    # Combine all dataframes
    combined_fairness = pd.concat(all_fairness_dfs, ignore_index=True) if all_fairness_dfs else pd.DataFrame()
    combined_leakage = pd.concat(all_leakage_dfs, ignore_index=True) if all_leakage_dfs else pd.DataFrame()
    combined_intrinsic = pd.concat(all_intrinsic_dfs, ignore_index=True) if all_intrinsic_dfs else pd.DataFrame()
    
    # Create comprehensive table with language information
    return create_comprehensive_multilingual_table(combined_fairness, combined_leakage, combined_intrinsic)

def create_comprehensive_multilingual_table(fairness_df, leakage_df, intrinsic_df):
    """
    Create comprehensive LaTeX table with fairness and leakage metrics across languages (no intrinsic bias).
    """
    if fairness_df.empty and leakage_df.empty:
        return "No data available for multilingual comprehensive table generation."
    
    # Merge dataframes on experimental conditions including language
    merge_cols = ['language', 'method', 'discriminators', 'masked', 'balanced']
    
    # Start with fairness results as base
    if not fairness_df.empty:
        combined_df = fairness_df[merge_cols + ['accuracy_gap', 'f1_gap', 'eq_odds_gap']].copy()
    else:
        combined_df = pd.DataFrame()
    
    # Add leakage results
    if not leakage_df.empty:
        leakage_subset = leakage_df[merge_cols + ['test_leakage', 'test_f1']].copy()
        if not combined_df.empty:
            combined_df = pd.merge(combined_df, leakage_subset, on=merge_cols, how='outer')
        else:
            combined_df = leakage_subset
    
    if combined_df.empty:
        return "No data available for multilingual comprehensive table."
    
    # Sort by language, method, then other parameters
    df_sorted = combined_df.sort_values(['language', 'method', 'discriminators', 'masked', 'balanced'])
    
    latex_lines = []
    
    # Table header - removed Bias ES column
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Multilingual Comprehensive Bias Results}")
    latex_lines.append("\\label{tab:multilingual_comprehensive}")
    latex_lines.append("\\begin{tabular}{lcccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Lang & Method & D & M & Acc Gap & F1 Gap & EqOdds & Test Leak \\\\")
    latex_lines.append("\\midrule")
    
    # Table rows
    for _, row in df_sorted.iterrows():
        language = row['language'].upper()
        method = row['method'].replace('_', '\\_')
        discriminators = row['discriminators'] if row['discriminators'] > 0 else '-'
        masked = 'Y' if row['masked'] else 'N'
        
        # Format numbers to 3 decimal places
        acc_gap = f"{row['accuracy_gap']:.3f}" if not pd.isna(row.get('accuracy_gap', np.nan)) else '-'
        f1_gap = f"{row['f1_gap']:.3f}" if not pd.isna(row.get('f1_gap', np.nan)) else '-'
        eq_odds_gap = f"{row['eq_odds_gap']:.3f}" if not pd.isna(row.get('eq_odds_gap', np.nan)) else '-'
        test_leakage = f"{row['test_leakage']:.3f}" if not pd.isna(row.get('test_leakage', np.nan)) else '-'
        
        latex_lines.append(f"{language} & {method} & {discriminators} & {masked} & {acc_gap} & {f1_gap} & {eq_odds_gap} & {test_leakage} \\\\")
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return '\n'.join(latex_lines)

def main():
    """
    Main function to automatically process all available languages and generate LaTeX tables.
    """
    base_results_dir = "../results/adv_debias/"
    
    print("MULTILINGUAL BIAS RESULTS TO LATEX")
    print("=" * 60)
    
    # Find all available languages
    languages = find_available_languages(base_results_dir)
    
    if not languages:
        print("❌ No language directories found!")
        print(f"Please check that results exist in: {base_results_dir}")
        return None
    
    print(f"🌍 Found {len(languages)} languages: {', '.join(languages)}")
    print("=" * 60)
    
    # Collect results from all languages
    all_fairness_dfs = []
    all_leakage_dfs = []
    all_intrinsic_dfs = []
    processed_languages = []
    all_saved_files = {}
    
    for language in languages:
        results_dir = f"{base_results_dir}/{language}/"
        
        fairness_df, leakage_df, intrinsic_df = load_results_for_language(results_dir, language)
        
        # Only keep languages that have at least some results
        if not fairness_df.empty or not leakage_df.empty or not intrinsic_df.empty:
            processed_languages.append(language)
            
            if not fairness_df.empty:
                all_fairness_dfs.append(fairness_df)
            if not leakage_df.empty:
                all_leakage_dfs.append(leakage_df)
            if not intrinsic_df.empty:
                all_intrinsic_dfs.append(intrinsic_df)
        
        print()  # Empty line for readability
    
    if not processed_languages:
        print("❌ No results found in any language directory!")
        return None
    
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed {len(processed_languages)} languages: {', '.join(processed_languages)}")
    print(f"📈 Total fairness results: {sum(len(df) for df in all_fairness_dfs)}")
    print(f"🔍 Total leakage results: {sum(len(df) for df in all_leakage_dfs)}")
    print(f"🧠 Total intrinsic results: {sum(len(df) for df in all_intrinsic_dfs)}")
    
    # Generate LaTeX tables
    print("\n" + "=" * 60)
    print("GENERATING LATEX TABLES")
    print("=" * 60)
    
    # Process each language individually and save in language-specific folders
    for language in processed_languages:
        print(f"\n🔤 Processing {language.upper()} tables...")
        
        # Find the dataframes for this language
        lang_fairness = next((df[df['language'] == language] for df in all_fairness_dfs if not df.empty and 'language' in df.columns and language in df['language'].values), pd.DataFrame())
        lang_leakage = next((df[df['language'] == language] for df in all_leakage_dfs if not df.empty and 'language' in df.columns and language in df['language'].values), pd.DataFrame())
        lang_intrinsic = next((df[df['language'] == language] for df in all_intrinsic_dfs if not df.empty and 'language' in df.columns and language in df['language'].values), pd.DataFrame())
        
        # Create tables for this language
        lang_tables = {}
        
        if not lang_fairness.empty:
            lang_tables[f'{language}_fairness'] = create_fairness_latex_table(lang_fairness, language=language)
            lang_tables[f'{language}_equalized_odds'] = create_equalized_odds_detailed_table(lang_fairness, language=language)
            print(f"  ✓ Created fairness and equalized odds tables")
        
        if not lang_leakage.empty:
            lang_tables[f'{language}_leakage'] = create_leakage_latex_table(lang_leakage, language=language)
            print(f"  ✓ Created leakage table")
        
        if not lang_intrinsic.empty:
            lang_tables[f'{language}_intrinsic'] = create_intrinsic_latex_table(
                lang_intrinsic, language=language, show_details=False)
            print(f"  ✓ Created intrinsic bias table")
        
        # Comprehensive table for this language (fairness + leakage only)
        if not lang_fairness.empty or not lang_leakage.empty:
            lang_tables[f'{language}_comprehensive'] = create_comprehensive_latex_table(
                lang_fairness, lang_leakage, pd.DataFrame(), 
                language=language, show_details=False)
            print(f"  ✓ Created comprehensive table")
        
        # Save tables in language-specific directory
        if lang_tables:
            saved_files = save_latex_tables(lang_tables, output_dir="../results/", language=language)
            all_saved_files.update(saved_files)
            
            print(f"  💾 Saved {len(lang_tables)} tables to: ../results/adv_debias/{language}/")
            for table_name in lang_tables.keys():
                table_type = table_name.replace(f'{language}_', '').replace('_', ' ').title()
                print(f"    - {table_type}")
        else:
            print(f"  ⚠️  No tables generated for {language}")
    
    # Create and save multilingual comparison tables if we have multiple languages
    if len(processed_languages) > 1:
        print(f"\n🌐 Creating multilingual comparison tables...")
        
        # Combine all results
        combined_fairness = pd.concat(all_fairness_dfs, ignore_index=True) if all_fairness_dfs else pd.DataFrame()
        combined_leakage = pd.concat(all_leakage_dfs, ignore_index=True) if all_leakage_dfs else pd.DataFrame()
        combined_intrinsic = pd.concat(all_intrinsic_dfs, ignore_index=True) if all_intrinsic_dfs else pd.DataFrame()
        
        # Multilingual comparison tables using existing functions with multilingual=True
        multilingual_tables = {}
        
        if not combined_fairness.empty:
            multilingual_tables['multilingual_fairness'] = create_fairness_latex_table(combined_fairness, multilingual=True)
            print(f"  ✓ Created multilingual fairness table")
        
        if not combined_leakage.empty:
            multilingual_tables['multilingual_leakage'] = create_leakage_latex_table(combined_leakage, multilingual=True)
            print(f"  ✓ Created multilingual leakage table")
        
        if not combined_intrinsic.empty:
            multilingual_tables['multilingual_intrinsic'] = create_intrinsic_latex_table(
                combined_intrinsic, multilingual=True, show_details=False)
            print(f"  ✓ Created multilingual intrinsic table")
        
        # Comprehensive multilingual table
        if not combined_fairness.empty or not combined_leakage.empty:
            multilingual_tables['multilingual_comprehensive'] = create_comprehensive_latex_table(
                combined_fairness, combined_leakage, pd.DataFrame(), 
                multilingual=True, show_details=False)
            print(f"  ✓ Created multilingual comprehensive table")
        
        # Save multilingual tables in general results directory
        if multilingual_tables:
            multilingual_saved_files = save_latex_tables(multilingual_tables, output_dir="../results/")
            all_saved_files.update(multilingual_saved_files)
            
            print(f"  💾 Saved {len(multilingual_tables)} multilingual tables to: ../results/")
            for table_name in multilingual_tables.keys():
                table_type = table_name.replace('multilingual_', '').replace('_', ' ').title()
                print(f"    - {table_type}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    # Count tables by language
    lang_table_count = {}
    multilingual_table_count = 0
    
    for file_path in all_saved_files.values():
        if 'multilingual_' in file_path:
            multilingual_table_count += 1
        else:
            for lang in processed_languages:
                if f'/adv_debias/{lang}/' in file_path:
                    lang_table_count[lang] = lang_table_count.get(lang, 0) + 1
                    break
    
    print(f"✅ Successfully generated tables for {len(processed_languages)} languages")
    
    for language in processed_languages:
        count = lang_table_count.get(language, 0) // 2  # Divide by 2 because we save both .tex and .txt
        print(f"   📁 {language.upper()}: {count} tables saved in ../results/adv_debias/{language}/")
    
    if multilingual_table_count > 0:
        count = multilingual_table_count // 2  # Divide by 2 because we save both .tex and .txt
        print(f"   🌐 MULTILINGUAL: {count} tables saved in ../results/")
    
    print(f"\n📄 Total files saved: {len(all_saved_files)}")
    
    # Return combined results for further analysis
    if len(processed_languages) == 1:
        return (all_fairness_dfs[0] if all_fairness_dfs else pd.DataFrame(),
                all_leakage_dfs[0] if all_leakage_dfs else pd.DataFrame(),
                all_intrinsic_dfs[0] if all_intrinsic_dfs else pd.DataFrame())
    else:
        return (pd.concat(all_fairness_dfs, ignore_index=True) if all_fairness_dfs else pd.DataFrame(),
                pd.concat(all_leakage_dfs, ignore_index=True) if all_leakage_dfs else pd.DataFrame(),
                pd.concat(all_intrinsic_dfs, ignore_index=True) if all_intrinsic_dfs else pd.DataFrame())

# Run the script
if __name__ == "__main__":
    results = main()
    if results is not None:
        fairness_df, leakage_df, intrinsic_df = results
        print(f"\n🎯 Final summary: {len(fairness_df)} fairness, {len(leakage_df)} leakage, {len(intrinsic_df)} intrinsic results loaded.")
    else:
        print("❌ Script completed but no results were processed.")