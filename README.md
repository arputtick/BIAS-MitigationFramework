# The BIAS Mitigation Framework

This work is part of the Europe Horizon project BIAS funded by the European Commission, and has received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI).
All work from the BIAS Project: https://github.com/BFH-AMI/BIAS 

## Overview
This repository contains code for conducting adversarial bias mitigation experiments on BERT and RoBERTa models in a variety of languages, focusing on reducing gender bias in occupation prediction tasks. The code supports both BERT and RoBERTa architectures and includes tools for intrinsic bias evaluation using WEAT/SEAT/LPBS tests.

The chief components of this repository include:
- Adversarial_experiments.ipynb: Main notebook for running bias mitigation experiments.
- BiasBios_Translation.ipynb: Notebook for translating the BiasBios dataset into new languages.
- profession_analysis.py: Script for analyzing profession-level TPR gaps and generating LaTeX tables.
- Secure configuration setup for managing API keys and credentials.

The main methodds implemented are:
- Pretrained Model Evaluation: Assessing bias in pretrained models without additional training.
- Standard Fine-tuning: Supervised learning on occupation prediction tasks.
- Adversarial Training: Bias mitigation through adversarial debiasing during training.
- Masking and balancing techniques to improve fairness: Removing gender pronouns and names from input text and undersampling majority groups to create balanced datasets.

The models are fine-tuned and evaluated on the BiasBios dataset, with comprehensive metrics including accuracy gaps, F1 gaps, and average TPR gaps. The repository also includes a translation pipeline for expanding the dataset to new languages using DeepL and Google Cloud Translation APIs.

In addition to adversarial training, the repository provides tools for intrinsic bias evaluation using WEAT/SEAT/LPBS tests, allowing researchers to assess the effectiveness of bias mitigation strategies.
## Related Papers
Adversarial debiasing is based on the following papers:

[1] X. Han, T. Baldwin, and T. Cohn, “Towards Equal Opportunity Fairness through Adversarial Learning,” May 15, 2022, arXiv: arXiv:2203.06317. doi: 10.48550/arXiv.2203.06317.

[2] Xudong Han, Timothy Baldwin, and Trevor Cohn. 2021. Diverse Adversaries for Mitigating Bias in Training. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2760–2765, Online. Association for Computational Linguistics.

The implelementation in [1] and [2] has been extended to multilingual BERT and RoBERTa models, with additional features such as secure configuration management, comprehensive fairness metrics, and a translation pipeline for dataset expansion.

WEAT and LPBS bias evaluation is based on the following papers:

[3] Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan. "Semantics derived automatically from language corpora contain human-like biases." Science 356.6334 (2017): 183-186.

[4] Kurita, Keita, et al. "Measuring bias in contextualized word representations." arXiv preprint arXiv:1906.07337 (2019).

Both methods have been adapted for BERT and RoBERTa multilingual evaluation in this repository. Additional tests and templates have been created for various languages to facilitate comprehensive bias assessment. These are based on research carried out in the BIAS project.

The BiasBios dataset used for fine-tuning and evaluation is described in the following paper:

[5] De-Arteaga, Maria, et al. "Bias in bios: A case study of semantic representation bias in a high-stakes setting." Proceedings of the 2019 Conference on Fairness, Accountability, and Transparency. 2019.

# Multilingual Bias Mitigation in BERT and RoBERTa Models

This repository contains code for adversarial bias mitigation experiments on multilingual BERT models, with support for both BERT and RoBERTa architectures.

## Features

- **Multilingual Support**: Works with BERT models for multiple languages (English, German, Norwegian, Italian, Turkish, Dutch) and RoBERTa models (Icelandic)
- **Adversarial Debiasing**: Implements adversarial training to reduce gender bias in occupation prediction
- **Intrinsic Bias Evaluation**: WEAT/SEAT/LPBS bias testing on fine-tuned models
- **Comprehensive Metrics**: Fairness evaluation including accuracy gaps, F1 gaps, and average TPR gaps
- **Secure Configuration**: API keys and credentials managed securely for public repositories

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up secure configuration
python setup_secure_config.py
```

### 2. Configure API Keys

Edit `config/credentials.json` with your API keys:
```json
{
    "openai": {"api_key": "your_key_here"},
    "voyageai": {"api_key": "your_key_here"},
    "deepl": {"api_key": "your_key_here"}
}
```

See [SECURITY.md](SECURITY.md) for detailed setup instructions.

### 3. Run Experiments

Open and run the main notebook:
```bash
jupyter notebook src/Adversarial_Experiments.ipynb
```

## Running Experiments

### Experiment Types

The main experiments are conducted in `src/Adversarial_Experiments.ipynb` which supports three training approaches:

#### 1. Pretrained Model Evaluation
- **Purpose**: Evaluate bias in pretrained models without additional training
- **Configuration**: Set `pretrained = True` in the notebook
- **Languages**: English (BERT), Icelandic (RoBERTa), German, Norwegian, Italian, Turkish, Dutch
- **Outputs**: Intrinsic bias metrics (WEAT/SEAT/LPBS), fairness evaluation on BiasBios dataset

#### 2. Standard Fine-tuning
- **Purpose**: Standard supervised learning on occupation prediction task
- **Configuration**: Set `pretrained = False`, `adversarial_training = False`
- **Process**: Fine-tunes the model on BiasBios dataset for occupation classification
- **Outputs**: Model checkpoints, bias evaluation, fairness metrics

#### 3. Adversarial Training
- **Purpose**: Bias mitigation through adversarial debiasing during training
- **Configuration**: Set `pretrained = True`, `adversarial_training = False`
- **Process**: Trains main classifier while adversarially training against gender discrimination
- **Parameters**: 
  - `LAMBDA`: Adversarial loss weight (0.8 recommended)
  - `diff_LAMBDA`: Weight of differential loss term for multiple discriminators
- **Outputs**: Debiased model, improved fairness metrics

### Model Configuration

The notebook automatically detects and loads the appropriate model architecture from utils.model_config.py:

```python
# Language-specific model loading
model_configs = {
    "en": "bert-base-uncased",
    "de": "dbmdz/bert-base-german-cased", 
    "nb": "NbAiLab/nb-bert-base",
    "it": "dbmdz/bert-base-italian-uncased",
    "tk": "dbmdz/bert-base-turkish-uncased",
    "nl": "GroNLP/bert-base-dutch-cased",
    "is": "mideind/IceBERT" # This is a RoBERTa-base model
    }
```

For Icelandic, the system automatically switches to RoBERTa tokenizer and model handling.

## Results and Outputs

### LaTeX Table Generation

The main output of experiments are formatted LaTeX tables ready for academic papers. Run the table generation script:

```bash
python src/results_to_latex.py
```

This automatically processes all available results and generates both language-specific and multilingual comparison tables.

### Generated LaTeX Tables

#### Language-Specific Tables
Located in `results/adv_debias/{language}/` (both `.tex` and `.txt` formats):

**1. Fairness Table (`{language}_fairness.tex`)**
- **Accuracy by Gender**: F (Female) and M (Male) accuracy scores
- **Accuracy Gap**: |Acc_M - Acc_F| (lower = more fair)
- **F1 by Gender**: F1 scores for female and male examples
- **F1 Gap**: |F1_M - F1_F| (lower = more fair)
- **Average TPR Gap**: True Positive Rate gap across occupations (more robust than equalized odds)

**2. Leakage Table (`{language}_leakage.tex`)**
- **Test Leakage**: Accuracy of adversarial discriminator (lower = less gender leakage)
- **Test F1**: F1 score of gender prediction (lower = better bias mitigation)
- Compares: Pretrained → Standard → Adversarial training progression

**3. Intrinsic Bias Table (`{language}_intrinsic.tex`)**
- **WEAT Tests**: Effect sizes for word association tests (e.g., Career vs Family)
- **LPBS Tests**: Log probability bias scores for template-based evaluation
- **Statistical Significance**: P-values for bias measurements

**4. Comprehensive Table (`{language}_comprehensive.tex`)**
- **Combined View**: Fairness + Leakage metrics in one table
- **Training Methods**: Pretrained, Standard Fine-tuning, Adversarial Training
- **Key Metrics**: Accuracy gaps, TPR gaps, discriminator performance

#### Multilingual Comparison Tables
Located in `results/` (for cross-language analysis):

**1. Multilingual Fairness (`multilingual_fairness.tex`)**
- Side-by-side comparison of fairness metrics across all languages
- Identifies which languages show more/less bias
- Method comparison (Standard vs Adversarial) per language

**2. Multilingual Comprehensive (`multilingual_comprehensive.tex`)**
- Complete cross-language bias mitigation results
- Language column for easy comparison
- Effectiveness of adversarial training by language

### Table Structure Example

```latex
\begin{tabular}{lcccc}
\toprule
Method & Acc F & Acc M & Acc Gap & Avg TPR Gap \\
\midrule
Pretrained & 0.82 & 0.78 & 0.04 & 0.15 \\
Standard & 0.85 & 0.83 & 0.02 & 0.08 \\
Adversarial & 0.84 & 0.83 & 0.01 & 0.03 \\
\bottomrule
\end{tabular}
```

### Interpreting LaTeX Tables

#### Fairness Metrics (Lower = Better)
- **Accuracy Gap < 0.02**: Excellent fairness
- **Accuracy Gap 0.02-0.05**: Good fairness  
- **Accuracy Gap > 0.05**: Concerning bias
- **TPR Gap < 0.05**: Low bias across occupations

#### Leakage Metrics (Lower = Better)
- **Test Leakage < 0.6**: Strong bias mitigation
- **Test Leakage 0.6-0.8**: Moderate mitigation
- **Test Leakage > 0.8**: Minimal bias reduction

#### Expected Results Pattern
1. **Pretrained**: Highest bias (largest gaps)
2. **Standard Training**: Reduced bias but still present
3. **Adversarial Training**: Lowest bias (smallest gaps, lowest leakage)

#### Statistical Interpretation
- **WEAT Effect Size**: Cohen's d, values > 0.8 indicate strong bias
- **TPR Gap**: More robust than equalized odds for small sample sizes
- **p-values**: < 0.05 indicates statistically significant bias

## Adding New Languages and WEAT Tests

### Adding a New Language

#### 1. Prepare Data Structure
```bash
mkdir -p data/wordlists/{language_code}/WEAT
mkdir -p data/wordlists/{language_code}/SEAT
mkdir -p data/wordlists/{language_code}/LPBS
```

#### 2. Add WEAT Word Lists
Create text files in `data/wordlists/{language_code}/WEAT/`:
- `WEAT_1.txt` - Career vs Family (male-female bias)
- `WEAT_7.txt` - Math vs Arts (male-female bias)
- `WEAT_8.txt` - Science vs Arts (male-female bias)

Format example:
```
# Target words (group A)
programmer, engineer, scientist
# Target words (group B)  
nurse, teacher, librarian
# Attribute words (group A)
male, man, boy
# Attribute words (group B)
female, woman, girl
```

#### 3. Add SEAT Templates
Create JSONL files in `data/wordlists/{language_code}/SEAT/`:
```json
{"template": "TEMPLATE_1", "bias_type": "gender_occupation", "target_A": ["engineer", "programmer"], "target_B": ["nurse", "teacher"], "attribute_A": ["he", "his"], "attribute_B": ["she", "her"]}
```

#### 4. Update Model Configuration
Add language to `src/utils/model_config.py`:
```python
model_types = {
    'your_language': 'your-bert-model-name'
}
```

#### 5. Configure Experiment
Update language list in notebooks:
```python
languages = ['en', 'de', 'no', 'it', 'tk', 'nl', 'is', 'your_language']
```

### Adding New WEAT Tests

#### 1. Create Word Lists
Add to `data/wordlists/{language}/WEAT/NEW_TEST.txt`:
```
# Define your word categories
# Target A: occupation/concept group 1
# Target B: occupation/concept group 2  
# Attribute A: stereotypically associated attributes
# Attribute B: counter-stereotypical attributes
```

#### 2. Update Test Configuration
Modify `src/bias_metrics/config.py` to include new tests:
```python
WEAT_TESTS = {
    'NEW_TEST': {
        'targets_A': 'target_words_A',
        'targets_B': 'target_words_B', 
        'attributes_A': 'attr_words_A',
        'attributes_B': 'attr_words_B'
    }
}
```

## Translation Pipeline

### Using the Translation Notebook

The `src/BiasBios_Translation.ipynb` notebook enables translation of the BiasBios dataset to new languages.

#### Prerequisites
1. **Configure APIs**: Set up DeepL and/or Google Cloud Translation APIs in `config/credentials.json`
2. **Install dependencies**: Ensure `deepl` and `google-cloud-translate` packages are installed

#### Translation Process

1. **Open the notebook**:
   ```bash
   jupyter notebook src/BiasBios_Translation.ipynb
   ```

2. **Configure target language**:
   ```python
   target_language = 'DE'  # ISO language code
   masked = True           # Translate masked or unmasked data
   ```

3. **Monitor translation costs**:
   - The notebook automatically counts characters and estimates costs
   - DeepL Pro: ~€20 per 1M characters
   - Google Translate: ~$20 per 1M characters

4. **Run translation**:
   - For most languages: Uses DeepL API (better quality)
   - For Icelandic: Uses Google Translate (DeepL doesn't support Icelandic)
   - Includes progress monitoring and error handling

#### Supported Translation Services

- **DeepL**: Higher quality, supports DE, NL, IT, NO, etc.
- **Google Cloud**: Broader language support, includes Icelandic
- **Automatic switching**: Notebook automatically chooses appropriate service

#### Output
Translated datasets are saved to:
```
data/bios/{language_code}/
├── train_split_balanced.csv
├── test_split_balanced.csv  
├── train_split_balanced_masked.csv
└── test_split_balanced_masked.csv
```

#### Character Limits and Costs
The notebook provides detailed cost estimation:
```
=== CHARACTER COUNT ANALYSIS ===
Train: 1,234,567 characters (€24.69 estimated)
Test: 234,567 characters (€4.69 estimated)  
Total: €29.38 estimated cost
```

## Repository Structure

```
├── config/                    # Secure configuration templates
├── data/                      # Datasets and wordlists  
├── src/                       # Source code
│   ├── bias_metrics/          # Bias evaluation metrics
│   ├── utils/                 # Utility functions
│   ├── Adversarial_Experiments.ipynb  # Main experiment notebook
│   └── ...
├── results/                   # Experiment outputs
├── requirements.txt           # Python dependencies
├── SECURITY.md               # Security and configuration guide
└── setup_secure_config.py    # Configuration setup script
```

## Best Practices and Troubleshooting

### Memory Management
- **Large Models**: RoBERTa and BERT models require significant GPU memory
- **Batch Size**: Reduce `batch_size` if encountering CUDA out-of-memory errors
- **Gradient Accumulation**: Use `gradient_accumulation_steps` for effective large batch training

### Model Loading Issues
- **Architecture Mismatch**: The system auto-detects BERT vs RoBERTa - ensure correct model names
- **Missing Weights**: If resuming training, ensure model checkpoint paths are correct
- **Tokenizer Compatibility**: BERT and RoBERTa use different tokenizers (auto-handled)

### Bias Evaluation
- **Small Sample Sizes**: Use average TPR gap instead of equalized odds for statistical reliability
- **Missing Tests**: Ensure WEAT/SEAT files exist for target language before evaluation
- **API Limits**: Monitor DeepL/OpenAI usage limits for embedding-based evaluations

### Translation
- **API Keys**: Ensure credentials are properly configured in `config/credentials.json`
- **Character Limits**: DeepL has monthly character limits - monitor usage
- **Language Support**: Use Google Translate for languages not supported by DeepL

### Common Issues
1. **Model Not Found**: Check model name in `language_models` dictionary
2. **CUDA Errors**: Reduce batch size or use CPU for smaller experiments  
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Permission Errors**: Ensure write access to `results/` directory

## Security

🛡️ This repository is configured for public sharing:
- API keys are never committed to git
- Credentials are loaded from secure config files
- Sensitive files are properly ignored

See [SECURITY.md](SECURITY.md) for complete security setup instructions.
# Profession-Level TPR Gap Analysis Script

This script analyzes existing evaluation results to extract profession-level True Positive Rate (TPR) gaps and generates LaTeX tables for academic papers.

## Overview

The script processes JSON results files generated by the `group_evaluation` function and creates comprehensive tables showing:

- **Top 3 professions favoring females** (largest negative TPR gaps)
- **Top 3 professions with smallest bias** (TPR gaps closest to zero) 
- **Top 3 professions favoring males** (largest positive TPR gaps)

## Usage

### Basic Usage

```bash
cd src/
python profession_analysis.py
```

This will:
- Auto-detect all available languages in `../results/adv_debias/`
- Load profession names from `../data/bios/`
- Generate LaTeX tables in `../results/profession_analysis/`

### Advanced Usage

```bash
# Process specific languages only
python profession_analysis.py --languages en,de,is

# Specify custom directories  
python profession_analysis.py \
    --results_dir /path/to/results/adv_debias/ \
    --data_path /path/to/data/bios/ \
    --output_dir /path/to/output/

# Get help
python profession_analysis.py --help
```

### Command Line Options

- `--results_dir`: Directory containing results files (default: `../results/adv_debias/`)
- `--languages`: Comma-separated list of languages to process (default: auto-detect all)
- `--data_path`: Path to dataset files for profession names (default: `../data/bios/`)
- `--output_dir`: Output directory for generated tables (default: `../results/profession_analysis/`)

## Input Requirements

### Results File Format

The script expects JSON files containing evaluation results with these keys:
- `Group 0 confusion matrix`: 2D array for female performance
- `Group 1 confusion matrix`: 2D array for male performance  
- `class_labels`: List of profession indices
- `num_classes`: Number of professions

Example structure:
```json
{
  "Group 0 confusion matrix": [[45, 2, 1], [3, 38, 4], [1, 5, 42]],
  "Group 1 confusion matrix": [[42, 5, 2], [4, 35, 6], [2, 7, 38]], 
  "class_labels": [0, 1, 2],
  "num_classes": 3
}
```

### Directory Structure

Expected input structure:
```
results/adv_debias/
├── en/
│   ├── bert_standard_balanced_masked_results.json
│   ├── bert_adversarial_balanced_masked_results.json
│   └── ...
├── de/
│   ├── bert_standard_balanced_masked_results.json
│   └── ...
└── is/
    ├── roberta_standard_balanced_masked_results.json
    └── ...
```

## Output

### Generated Files

For each results file, the script generates:
- `{original_name}_profession_tpr_gaps.tex` - LaTeX table code
- `{original_name}_profession_tpr_gaps.txt` - Same content as text file

### Output Structure

```
results/profession_analysis/
├── en/
│   ├── bert_standard_balanced_masked_results_profession_tpr_gaps.tex
│   ├── bert_standard_balanced_masked_results_profession_tpr_gaps.txt
│   └── ...
├── de/
│   └── ...
└── is/
    └── ...
```

### LaTeX Table Format

The generated tables include:
- **Category**: Female Favored / Smallest Gap / Male Favored
- **Profession**: Name of the profession 
- **Female TPR**: True Positive Rate for females
- **Male TPR**: True Positive Rate for males
- **Gap (M-F)**: Male TPR - Female TPR (signed difference)

## Interpretation

### TPR Gap Values
- **Positive gaps**: Model performs better for males in that profession
- **Negative gaps**: Model performs better for females in that profession  
- **Near-zero gaps**: Fair/equal performance across genders

### Categories
- **Female Favored**: Professions where model shows female bias (most negative gaps)
- **Male Favored**: Professions where model shows male bias (most positive gaps)  
- **Smallest Gap**: Most fair professions (least biased predictions)

## Example Output

```
🔤 Processing EN (3 files)
  ✓ Created tables for bert_standard_balanced_masked_results
    Female-favored: nurse, teacher, librarian
    Male-favored: engineer, programmer, surgeon
    Smallest gaps: accountant, lawyer, manager
```

## Integration with Papers

The generated LaTeX tables can be directly included in academic papers:

```latex
\input{results/profession_analysis/en/bert_standard_results_profession_tpr_gaps.tex}
```

## Troubleshooting

### Common Issues

1. **No results files found**: Check that `--results_dir` points to the correct location
2. **Could not load profession names**: Ensure dataset files exist in `--data_path`  
3. **Insufficient data warning**: Some professions may have too few examples for reliable analysis

### Requirements

- Python 3.6+
- pandas (for loading profession names)
- numpy (for confusion matrix calculations)
- Standard library: json, os, glob, argparse, pathlib

## Example Run

```bash
$ python profession_analysis.py --languages en,de,is

🔍 PROFESSION-LEVEL TPR GAP ANALYSIS
==================================================
Processing specified languages: ['en', 'de', 'is']

📋 Loading profession names from ../data/bios/
Loaded 28 profession names from ../data/bios/en/train_split_balanced.csv

📂 Searching for results files in ../results/adv_debias/
Found 2 results files for en: ['bert_standard_results.json', 'bert_adversarial_results.json']  
Found 2 results files for de: ['bert_standard_results.json', 'bert_adversarial_results.json']
Found 1 results files for is: ['roberta_standard_results.json']

✅ ANALYSIS COMPLETE  
Generated 5 profession analysis tables
```


