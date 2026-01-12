from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


def compute_multiclass_equalized_odds(y_true, y_pred, p_labels, unique_classes):
    """
    Compute equalized odds gap for multiclass classification.
    
    For each class, compute TPR and FPR for each protected group,
    then calculate the maximum gap across all classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        p_labels: Protected attribute labels
        unique_classes: List of unique class labels
        
    Returns:
        dict: Equalized odds metrics
    """
    # Split by protected groups
    g0_mask = (p_labels == 0)
    g1_mask = (p_labels == 1)
    
    g0_true = y_true[g0_mask]
    g0_pred = y_pred[g0_mask]
    g1_true = y_true[g1_mask]
    g1_pred = y_pred[g1_mask]
    
    tpr_gaps = []
    fpr_gaps = []
    
    class_metrics = {}
    
    for class_label in unique_classes:
        # For each class, treat it as positive vs all others (one-vs-rest)
        
        # Group 0 metrics for this class
        g0_tp = np.sum((g0_true == class_label) & (g0_pred == class_label))
        g0_fn = np.sum((g0_true == class_label) & (g0_pred != class_label))
        g0_fp = np.sum((g0_true != class_label) & (g0_pred == class_label))
        g0_tn = np.sum((g0_true != class_label) & (g0_pred != class_label))
        
        # Group 1 metrics for this class
        g1_tp = np.sum((g1_true == class_label) & (g1_pred == class_label))
        g1_fn = np.sum((g1_true == class_label) & (g1_pred != class_label))
        g1_fp = np.sum((g1_true != class_label) & (g1_pred == class_label))
        g1_tn = np.sum((g1_true != class_label) & (g1_pred != class_label))
        
        # Calculate TPR and FPR for each group
        g0_tpr = g0_tp / (g0_tp + g0_fn) if (g0_tp + g0_fn) > 0 else 0
        g0_fpr = g0_fp / (g0_fp + g0_tn) if (g0_fp + g0_tn) > 0 else 0
        
        g1_tpr = g1_tp / (g1_tp + g1_fn) if (g1_tp + g1_fn) > 0 else 0
        g1_fpr = g1_fp / (g1_fp + g1_tn) if (g1_fp + g1_tn) > 0 else 0
        
        # Calculate gaps for this class (Male - Female)
        tpr_gap = g1_tpr - g0_tpr  # Male - Female
        fpr_gap = g1_fpr - g0_fpr  # Male - Female
        
        tpr_gaps.append(tpr_gap)
        fpr_gaps.append(fpr_gap)
        
        # Store per-class metrics
        class_metrics[f"class_{class_label}_tpr_g0"] = g0_tpr
        class_metrics[f"class_{class_label}_tpr_g1"] = g1_tpr
        class_metrics[f"class_{class_label}_fpr_g0"] = g0_fpr
        class_metrics[f"class_{class_label}_fpr_g1"] = g1_fpr
        class_metrics[f"class_{class_label}_tpr_gap"] = tpr_gap
        class_metrics[f"class_{class_label}_fpr_gap"] = fpr_gap
    
    # Overall equalized odds gap is the maximum absolute gap across classes
    max_tpr_gap = max(abs(gap) for gap in tpr_gaps) if tpr_gaps else 0
    max_fpr_gap = max(abs(gap) for gap in fpr_gaps) if fpr_gaps else 0
    
    # Average gaps across classes (absolute values for summary, individual gaps remain signed)
    avg_tpr_gap = np.mean([abs(gap) for gap in tpr_gaps]) if tpr_gaps else 0
    avg_fpr_gap = np.mean([abs(gap) for gap in fpr_gaps]) if fpr_gaps else 0
    
    # Equalized odds violation is the maximum of absolute TPR and FPR gaps
    eq_odds_gap = max(max_tpr_gap, max_fpr_gap)
    
    results = {
        "eq_odds_gap": eq_odds_gap,
        "max_tpr_gap": max_tpr_gap,
        "max_fpr_gap": max_fpr_gap,
        "avg_tpr_gap": avg_tpr_gap,
        "avg_fpr_gap": avg_fpr_gap,
        **class_metrics
    }
    
    return results


def group_evaluation(preds, labels, p_labels, silence=True):
    """
    Evaluate fairness metrics for multi-class classification.
    
    Args:
        preds: Model predictions for main task
        labels: True labels for main task (can be multi-class)
        p_labels: Protected attribute labels (binary: 0 or 1)
        silence: Whether to print detailed results
    
    Returns:
        dict: Dictionary containing fairness metrics
    """
    preds = np.array(preds)
    labels = np.array(labels)
    p_labels = np.array(p_labels)

    p_set = set(p_labels)
    assert len(p_set) == 2, "Assuming binary protected attribute labels"

    # Split by protected groups
    g1_preds = preds[np.array(p_labels) == 1]
    g1_labels = labels[np.array(p_labels) == 1]

    g0_preds = preds[np.array(p_labels) == 0]
    g0_labels = labels[np.array(p_labels) == 0]

    # Calculate accuracies
    acc_0 = accuracy_score(g0_labels, g0_preds)
    acc_1 = accuracy_score(g1_labels, g1_preds)

    # Calculate F1 scores (macro average for multi-class)
    f1_0 = f1_score(g0_labels, g0_preds, average='macro', zero_division=0)
    f1_1 = f1_score(g1_labels, g1_preds, average='macro', zero_division=0)

    # Get unique classes for multi-class metrics
    unique_classes = np.unique(np.concatenate([labels]))
    num_classes = len(unique_classes)

    results = {
        "Accuracy_0": acc_0,
        "Accuracy_1": acc_1,
        "Accuracy_gap": acc_1 - acc_0,  # Male - Female
        "F1_macro_0": f1_0,
        "F1_macro_1": f1_1,
        "F1_gap": f1_1 - f1_0,  # Male - Female
        "Group 0 confusion matrix": confusion_matrix(g0_labels, g0_preds, labels=unique_classes).tolist(),
        "Group 1 confusion matrix": confusion_matrix(g1_labels, g1_preds, labels=unique_classes).tolist(),
        "num_classes": num_classes,
        "class_labels": unique_classes.tolist()
    }

    # Compute equalized odds gap (works for both binary and multi-class)
    eq_odds_results = compute_multiclass_equalized_odds(labels, preds, p_labels, unique_classes)
    results.update(eq_odds_results)

    # For binary classification, also calculate TPR/TNR (legacy metrics)
    if num_classes == 2:
        try:
            tn0, fp0, fn0, tp0 = confusion_matrix(g0_labels, g0_preds, labels=unique_classes).ravel()
            TPR0 = tp0/(fn0+tp0) if (fn0+tp0) > 0 else 0
            TNR0 = tn0/(fp0+tn0) if (fp0+tn0) > 0 else 0
        except ValueError:
            # Handle cases where one group might not have both classes
            TPR0, TNR0 = 0, 0

        try:
            tn1, fp1, fn1, tp1 = confusion_matrix(g1_labels, g1_preds, labels=unique_classes).ravel()
            TPR1 = tp1/(fn1+tp1) if (fn1+tp1) > 0 else 0
            TNR1 = tn1/(tn1+fp1) if (tn1+fp1) > 0 else 0
        except ValueError:
            TPR1, TNR1 = 0, 0

        results.update({
            "TPR_0": TPR0,
            "TPR_1": TPR1,
            "TNR_0": TNR0,
            "TNR_1": TNR1,
            "TPR_gap": TPR1 - TPR0,  # Male - Female
            "TNR_gap": TNR1 - TNR0  # Male - Female
        })

    # For multi-class, calculate per-class precision and recall
    else:
        # Calculate per-class metrics for each group
        from sklearn.metrics import precision_recall_fscore_support
        
        precision_0, recall_0, _, _ = precision_recall_fscore_support(
            g0_labels, g0_preds, labels=unique_classes, average=None, zero_division=0
        )
        precision_1, recall_1, _, _ = precision_recall_fscore_support(
            g1_labels, g1_preds, labels=unique_classes, average=None, zero_division=0
        )

        # Store per-class metrics
        for i, class_label in enumerate(unique_classes):
            results[f"precision_class_{class_label}_group_0"] = precision_0[i] if i < len(precision_0) else 0
            results[f"precision_class_{class_label}_group_1"] = precision_1[i] if i < len(precision_1) else 0
            results[f"recall_class_{class_label}_group_0"] = recall_0[i] if i < len(recall_0) else 0
            results[f"recall_class_{class_label}_group_1"] = recall_1[i] if i < len(recall_1) else 0
            
            # Calculate gaps (Male - Female)
            prec_gap = (
                (precision_1[i] if i < len(precision_1) else 0) -
                (precision_0[i] if i < len(precision_0) else 0)
            )
            recall_gap = (
                (recall_1[i] if i < len(recall_1) else 0) -
                (recall_0[i] if i < len(recall_0) else 0)
            )
            
            results[f"precision_gap_class_{class_label}"] = prec_gap
            results[f"recall_gap_class_{class_label}"] = recall_gap

    if not silence:
        print("=== FAIRNESS EVALUATION RESULTS ===")
        print(f"Number of classes: {num_classes}")
        print(f"Class labels: {unique_classes}")
        print(f"Group 0 size: {len(g0_labels)}, Group 1 size: {len(g1_labels)}")
        print()
        print("=== MAIN FAIRNESS METRICS ===")
        print(f"Accuracy Gap (M-F): {results['Accuracy_gap']:.4f}")
        print(f"F1 Macro Gap (M-F): {results['F1_gap']:.4f}")
        print(f"Equalized Odds Gap: {results['eq_odds_gap']:.4f}")
        print()
        print("=== DETAILED METRICS ===")
        print(f"Accuracy Group 0 (Female): {acc_0:.4f}")
        print(f"Accuracy Group 1 (Male): {acc_1:.4f}")
        print(f"F1 Macro Group 0 (Female): {f1_0:.4f}")
        print(f"F1 Macro Group 1 (Male): {f1_1:.4f}")
        print()
        print("=== EQUALIZED ODDS BREAKDOWN ===")
        print(f"Max TPR Gap: {results['max_tpr_gap']:.4f}")
        print(f"Max FPR Gap: {results['max_fpr_gap']:.4f}")
        print(f"Average TPR Gap (M-F): {results['avg_tpr_gap']:.4f}")
        print(f"Average FPR Gap (M-F): {results['avg_fpr_gap']:.4f}")
        
        if num_classes == 2:
            print()
            print("=== BINARY CLASSIFICATION METRICS ===")
            print(f"TPR Group 0 (Female): {results['TPR_0']:.4f}")
            print(f"TPR Group 1 (Male): {results['TPR_1']:.4f}")
            print(f"TPR Gap (M-F): {results['TPR_gap']:.4f}")
            print(f"TNR Group 0 (Female): {results['TNR_0']:.4f}")
            print(f"TNR Group 1 (Male): {results['TNR_1']:.4f}")
            print(f"TNR Gap (M-F): {results['TNR_gap']:.4f}")
        else:
            print()
            print("=== PER-CLASS EQUALIZED ODDS ===")
            for class_label in unique_classes:
                tpr_g0 = results.get(f"class_{class_label}_tpr_g0", 0)
                tpr_g1 = results.get(f"class_{class_label}_tpr_g1", 0)
                fpr_g0 = results.get(f"class_{class_label}_fpr_g0", 0)
                fpr_g1 = results.get(f"class_{class_label}_fpr_g1", 0)
                tpr_gap = results.get(f"class_{class_label}_tpr_gap", 0)
                fpr_gap = results.get(f"class_{class_label}_fpr_gap", 0)
                
                print(f"  Class {class_label}:")
                print(f"    TPR: Female={tpr_g0:.3f}, Male={tpr_g1:.3f}, Gap(M-F)={tpr_gap:.3f}")
                print(f"    FPR: Female={fpr_g0:.3f}, Male={fpr_g1:.3f}, Gap(M-F)={fpr_gap:.3f}")
            
            print()
            print("=== PER-CLASS PRECISION/RECALL ===")
            for class_label in unique_classes:
                prec_0 = results.get(f"precision_class_{class_label}_group_0", 0)
                prec_1 = results.get(f"precision_class_{class_label}_group_1", 0)
                recall_0 = results.get(f"recall_class_{class_label}_group_0", 0)
                recall_1 = results.get(f"recall_class_{class_label}_group_1", 0)
                prec_gap = results.get(f"precision_gap_class_{class_label}", 0)
                recall_gap = results.get(f"recall_gap_class_{class_label}", 0)
                
                print(f"  Class {class_label}:")
                print(f"    Precision: Female={prec_0:.3f}, Male={prec_1:.3f}, Gap(M-F)={prec_gap:.3f}")
                print(f"    Recall: Female={recall_0:.3f}, Male={recall_1:.3f}, Gap(M-F)={recall_gap:.3f}")

    return results


def leakage_evaluation(model, training_loader, test_loader, device):
    """
    Evaluate how much protected attribute information leaks through model representations.
    This function works for both binary and multi-class main tasks.
    """
    model.eval()

    train_hidden = []
    train_labels = []
    train_private_labels = []

    for batch in training_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['occupation_label'].to(device).long()               # main task labels
        p_tags = batch['label'].to(device).long() # private/auxiliary labels

        train_labels += list(tags.cpu().numpy())
        train_private_labels += list(p_tags.cpu().numpy())
        
        # Forward pass (HF models return ModelOutput)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
        
        # Hidden representations from model
        hs = outputs.hidden_states[-1][:, 0, :]  # shape: [batch_size, hidden_size], corresponds to [CLS]
        train_hidden.append(hs.detach().cpu().numpy())
    
    train_hidden = np.concatenate(train_hidden, 0)
    print(f"Training hidden shape: {train_hidden.shape}, Private labels: {len(train_private_labels)}")

    test_hidden = []
    test_labels = []
    test_private_labels = []

    for batch in test_loader:
         # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['occupation_label'].to(device).long()               # main task labels
        p_tags = batch['label'].to(device).long() # private/auxiliary labels

        test_labels += list(tags.cpu().numpy())
        test_private_labels += list(p_tags.cpu().numpy())
        
        # Forward pass (HF models return ModelOutput)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
        
        # Hidden representations from model
        hs = outputs.hidden_states[-1][:, 0, :]  # shape: [batch_size, hidden_size], corresponds to [CLS]
        test_hidden.append(hs.detach().cpu().numpy())
    
    test_hidden = np.concatenate(test_hidden, 0)

    # Train leakage classifier (protected attribute prediction from hidden states)
    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    biased_classifier.fit(train_hidden, train_private_labels)
    
    train_leakage = biased_classifier.score(train_hidden, train_private_labels)
    test_leakage = biased_classifier.score(test_hidden, test_private_labels)
    test_f1 = f1_score(test_private_labels, biased_classifier.predict(test_hidden), average='macro')
    
    # Handle binary vs multi-class for AUC calculation
    unique_private_labels = np.unique(train_private_labels)
    if len(unique_private_labels) == 2:
        test_roc_auc = roc_auc_score(test_private_labels, biased_classifier.decision_function(test_hidden))
        test_ap = average_precision_score(test_private_labels, biased_classifier.decision_function(test_hidden))
    else:
        # For multi-class, use probability scores if available
        try:
            # Try to get probability estimates
            prob_classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
            prob_classifier.fit(train_hidden, train_private_labels)
            test_probs = prob_classifier.predict_proba(test_hidden)
            test_roc_auc = roc_auc_score(test_private_labels, test_probs, multi_class='ovr', average='macro')
            test_ap = average_precision_score(test_private_labels, test_probs, average='macro')
        except:
            test_roc_auc = None
            test_ap = None

    print("=== LEAKAGE EVALUATION RESULTS ===")
    print(f"Train Leakage Accuracy: {train_leakage:.4f}")
    print(f"Test Leakage Accuracy: {test_leakage:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        print(f"Test Average Precision: {test_ap:.4f}")

    leakage_results = {
        "Train Leakage": train_leakage,
        "Test Leakage": test_leakage,
        "Test F1 Score": test_f1,
        "Confusion Matrix": confusion_matrix(test_private_labels, biased_classifier.predict(test_hidden)).tolist(),
        "num_private_classes": len(unique_private_labels)
    }
    
    if test_roc_auc is not None:
        leakage_results.update({
            "Test ROC AUC": test_roc_auc,
            "Test Average Precision": test_ap
        })

    return leakage_results


def analyze_profession_tpr_gaps(preds, labels, p_labels, profession_labels, profession_names, silence=True):
    """
    Analyze TPR gaps at the profession level to identify professions with largest gaps.
    
    Args:
        preds: Model predictions for main task
        labels: True labels for main task (profession indices)
        p_labels: Protected attribute labels (0=Female, 1=Male)
        profession_labels: List of profession indices
        profession_names: List of profession names corresponding to indices
        silence: Whether to print detailed results
        
    Returns:
        dict: Dictionary containing profession-level TPR analysis
    """
    preds = np.array(preds)
    labels = np.array(labels)
    p_labels = np.array(p_labels)
    
    profession_tpr_data = []
    
    for prof_idx in np.unique(labels):
        if prof_idx >= len(profession_names):
            continue
            
        prof_name = profession_names[prof_idx]
        
        # Get samples for this profession
        prof_mask = (labels == prof_idx)
        prof_preds = preds[prof_mask]
        prof_labels = labels[prof_mask]
        prof_p_labels = p_labels[prof_mask]
        
        if len(prof_p_labels) < 2:
            continue  # Skip professions with insufficient data
            
        # Split by gender
        female_mask = (prof_p_labels == 0)
        male_mask = (prof_p_labels == 1)
        
        if not np.any(female_mask) or not np.any(male_mask):
            continue  # Skip if missing either gender
            
        # Calculate TPR for each gender (True Positive Rate = Recall for this class)
        female_correct = np.sum((prof_labels[female_mask] == prof_idx) & (prof_preds[female_mask] == prof_idx))
        female_total = np.sum(prof_labels[female_mask] == prof_idx)
        female_tpr = female_correct / female_total if female_total > 0 else 0
        
        male_correct = np.sum((prof_labels[male_mask] == prof_idx) & (prof_preds[male_mask] == prof_idx))
        male_total = np.sum(prof_labels[male_mask] == prof_idx)
        male_tpr = male_correct / male_total if male_total > 0 else 0
        
        # Calculate gap (Male - Female)
        tpr_gap = male_tpr - female_tpr
        
        profession_tpr_data.append({
            'profession': prof_name,
            'profession_idx': prof_idx,
            'female_tpr': female_tpr,
            'male_tpr': male_tpr,
            'tpr_gap': tpr_gap,
            'female_count': int(np.sum(female_mask)),
            'male_count': int(np.sum(male_mask))
        })
    
    # Sort by TPR gap
    profession_tpr_data.sort(key=lambda x: x['tpr_gap'])
    
    # Get top/bottom professions
    largest_female_favor = profession_tpr_data[:3]  # Most negative gaps (favor females)
    smallest_gaps = sorted(profession_tpr_data, key=lambda x: abs(x['tpr_gap']))[:3]
    largest_male_favor = profession_tpr_data[-3:][::-1]  # Most positive gaps (favor males)
    
    results = {
        'all_professions': profession_tpr_data,
        'largest_female_favor': largest_female_favor,
        'smallest_gaps': smallest_gaps,
        'largest_male_favor': largest_male_favor
    }
    
    if not silence:
        print("=== PROFESSION-LEVEL TPR GAP ANALYSIS ===\n")
        
        print("TOP 3 PROFESSIONS FAVORING FEMALES (Most Negative Gaps):")
        for i, prof in enumerate(largest_female_favor, 1):
            print(f"{i}. {prof['profession']}: Gap={prof['tpr_gap']:.3f} "
                  f"(F:{prof['female_tpr']:.3f}, M:{prof['male_tpr']:.3f})")
        
        print("\nTOP 3 PROFESSIONS WITH SMALLEST GAPS:")
        for i, prof in enumerate(smallest_gaps, 1):
            print(f"{i}. {prof['profession']}: Gap={prof['tpr_gap']:.3f} "
                  f"(F:{prof['female_tpr']:.3f}, M:{prof['male_tpr']:.3f})")
        
        print("\nTOP 3 PROFESSIONS FAVORING MALES (Most Positive Gaps):")
        for i, prof in enumerate(largest_male_favor, 1):
            print(f"{i}. {prof['profession']}: Gap={prof['tpr_gap']:.3f} "
                  f"(F:{prof['female_tpr']:.3f}, M:{prof['male_tpr']:.3f})")
    
    return results