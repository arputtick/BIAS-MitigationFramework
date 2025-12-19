from pyexpat import model
import os,argparse,time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# from dataloaders.deep_moji import DeepMojiDataset
# from networks.deepmoji_sa import DeepMojiModel
from utils.discriminator import Discriminator


from tqdm import tqdm, trange
from utils.customized_loss import DiffLoss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.eval_metrices import group_evaluation, leakage_evaluation

from pathlib import Path, PureWindowsPath

import argparse


# convert fine-tuned model to masked language model for WEAT evaluation
from transformers import BertForMaskedLM, RobertaForMaskedLM

def convert_classification_to_mlm(classification_model, base_model_name="bert-base-uncased", model_type="bert"):
    """
    Convert a BertForSequenceClassification or RobertaForSequenceClassification model to MLM
    by transferring the backbone weights and adding a fresh MLM head
    """
    # Determine if it's RoBERTa or BERT based on model name or type
    is_roberta = model_type == "roberta" or "roberta" in base_model_name.lower() or "icebert" in base_model_name.lower()
    
    # Create a new MLM model with the same config
    if is_roberta:
        mlm_model = RobertaForMaskedLM.from_pretrained(base_model_name, output_hidden_states=True)
        backbone_prefix = 'roberta.'
    else:
        mlm_model = BertForMaskedLM.from_pretrained(base_model_name, output_hidden_states=True)
        backbone_prefix = 'bert.'
    
    # Transfer the backbone weights from your fine-tuned model
    mlm_state_dict = mlm_model.state_dict()
    classification_state_dict = classification_model.state_dict()
    
    # Copy backbone weights (everything except the classifier head)
    for name, param in classification_state_dict.items():
        if name.startswith(backbone_prefix):  # Only copy backbone layers, not classifier
            if name in mlm_state_dict:
                mlm_state_dict[name] = param
                print(f"Transferred: {name}")
    
    # Load the updated state dict
    mlm_model.load_state_dict(mlm_state_dict)
    
    return mlm_model

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, adv_optimizers, criterion, device, args):
    """"
    Train the discriminator to get a meaningful gradient
    """

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.train()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    
    for batch in iterator:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['occupation_label'].to(device).long()               # main task labels
        p_tags = batch['label'].to(device).long() # private/auxiliary labels

        with torch.no_grad():
            # Forward pass (HF models return ModelOutput)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags) 
            # Hidden representations from model
            hs = outputs.hidden_states[-1][:, 0, :].detach()  # shape: [batch_size, hidden_size], corresponds to [CLS]
        
        # iterate all discriminators
        for discriminator, adv_optimizer in zip(discriminators, adv_optimizers):
        
            adv_optimizer.zero_grad()

            adv_predictions = discriminator(hs)

        
            loss = criterion(adv_predictions, p_tags)

            # encrouge orthogonality
            if args.DL == True:
                # Get hidden representation.
                adv_hs_current = discriminator.hidden_representation(hs)
                for discriminator2 in discriminators:
                    if discriminator != discriminator2:
                        adv_hs = discriminator2.hidden_representation(hs)
                        # Calculate diff_loss
                        # should not include the current model
                        difference_loss = args.diff_LAMBDA * args.diff_loss(adv_hs_current, adv_hs)
                        loss = loss + difference_loss
                        
            loss.backward()
        
            adv_optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluate the discriminator
def adv_eval_epoch(model, discriminators, iterator, criterion, device, args):

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.eval()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    

    preds = {i:[] for i in range(args.n_discriminator)}
    labels = []
    private_labels = []

    with torch.no_grad():
        for batch in iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['occupation_label'].to(device).long()               # main task labels
            p_tags = batch['label'].to(device).long() # private/auxiliary labels

            # Forward pass (HF models return ModelOutput)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
            
            # Hidden representations from model
            hs = outputs.hidden_states[-1][:, 0, :]  # shape: [batch_size, hidden_size], corresponds to [CLS]
            
            # let discriminator make predictions
            for index, discriminator in enumerate(discriminators):
                adv_pred = discriminator(hs)
            
                loss = criterion(adv_pred, p_tags)
                            
                epoch_loss += loss.item()
            
                adv_predictions = adv_pred.detach().cpu()
                preds[index] += list(torch.argmax(adv_predictions, axis=1).numpy())


            tags = tags.cpu().numpy()

            labels += list(tags)
            
            private_labels += list(p_tags.cpu().numpy())
        
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

# train the main model with adv loss (rewritten for HF models)
def train_epoch(model, discriminators, iterator, optimizer, criterion, device, args):
    epoch_loss = 0.0
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()
        discriminator.GR = True  # activate gradient reversal layer

    for batch in iterator:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['occupation_label'].to(device).long()               # main task labels
        p_tags = batch['label'].to(device).long()  # private labels

        optimizer.zero_grad()

        # Forward pass (main model)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
        # print(outputs)
        predictions = outputs.logits
        loss = outputs.loss  # main classification loss

        # --- Adversarial training ---
        if args.adv:
            # Hidden representations from model
            hs = outputs.hidden_states[-1][:, 0, :]  # shape: [batch_size, hidden_size], corresponds to [CLS]

            for discriminator in discriminators:
                adv_predictions = discriminator(hs)
                adv_loss = criterion(input = adv_predictions, target = p_tags)
                loss = loss + adv_loss / len(discriminators)

        # Backpropagation + optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


# to evaluate the main model (rewrittnen for HF models)

def eval_main(model, iterator, device, args, criterion = torch.nn.CrossEntropyLoss()):
    epoch_loss = 0.0
    model.eval()
    
    preds = []
    labels = []
    private_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['occupation_label'].to(device).long()               # main task labels
            p_tags = batch['label'].to(device).float() # private/auxiliary labels

            # Forward pass (HF models return ModelOutput)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
            loss = outputs.loss
            logits = outputs.logits

            epoch_loss += loss.item()

            # Predictions + labels to CPU for collection
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            gold_labels = tags.cpu().numpy()

            preds.extend(predictions)
            labels.extend(gold_labels)
            private_labels.extend(p_tags.cpu().numpy())
    
    return (epoch_loss / len(iterator), preds, labels, private_labels)

def log_uniform(power_low, power_high):
    return np.power(10, np.random.uniform(power_low, power_high))


# After training, train a discriminator to see how much private information is encoded in the representation
def train_leakage_discriminator(model, iterator, device, args):
    model.eval()
    
    # Init discriminator
    discriminator = Discriminator(args, args.hidden_size, 2)
    discriminator = discriminator.to(device)
    
    # Init optimizer
    LEARNING_RATE = 0.001
    optimizer = Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=LEARNING_RATE)
    
    # Init criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in trange(20, desc="Leakage Discriminator Training"):
        epoch_loss = 0.0
        
        discriminator.train()
        
        for batch in iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['occupation_label'].to(device).long()               # main task labels
            p_tags = batch['label'].to(device).long()  # private labels

            optimizer.zero_grad()

            # Forward pass (HF models return ModelOutput)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
            
            # Hidden representations from model
            hs = outputs.hidden_states[-1][:, 0, :]  # shape: [batch_size, hidden_size], corresponds to [CLS]
            
            adv_predictions = discriminator(hs)
        
            loss = criterion(adv_predictions, p_tags)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(iterator)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    
    return discriminator


# Training and evaluation functions for BERT model with adversarial training
def train_and_evaluate(model, train_loader, test_loader, args, discriminators, optimizer, adv_optimizers, output_dir):
    """
    Train and evaluate the model with adversarial debiasing.
    
    Args:
        model: The main model to train
        train_loader: Training data loader
        test_loader: Test data loader
        args: Training arguments
        discriminators: List of discriminator models
        optimizer: Main model optimizer
        adv_optimizers: List of discriminator optimizers
        output_dir: Directory to save plots and results
    """
    num_epochs = args.num_epochs
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_type = args.experiment_type
    num_discriminators = args.n_discriminator
    balanced = args.balanced
    masked = args.masked
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs("models", exist_ok=True)

    # path to checkpoints (using forward slashes for cross-platform compatibility)
    main_model_path = "models/BERT_model_{}.pt".format(experiment_type)
    adv_model_paths = ["models/discriminator_{}_{}.pt".format(experiment_type, i) for i in range(num_discriminators)]


    best_loss, valid_preds, valid_labels, _ = eval_main(
                                                        model = model, 
                                                        iterator = test_loader, 
                                                        criterion = criterion, 
                                                        device = device, 
                                                        args = args
                                                        )

    best_acc = accuracy_score(valid_labels, valid_preds)
    torch.save(model.state_dict(), main_model_path) # save initial model

    best_epoch = -1
    main_val_accs = []
    main_train_accs = []
    disc_val_accs = {}

    for i in trange(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        print(f"Current Best Accuracy: {best_acc:.4f} at epoch {best_epoch} with loss {best_loss:.4f}")
        
        print("Training main model...")
        # Train main model
        train_epoch(
                    model = model, 
                    discriminators = discriminators, 
                    iterator = train_loader, 
                    optimizer = optimizer, 
                    criterion = criterion, 
                    device = device, 
                    args = args
                    )
        train_loss, train_preds, train_labels, _ = eval_main(
                                                            model = model, 
                                                            iterator = train_loader, 
                                                            criterion = criterion, 
                                                            device = device, 
                                                            args = args
                                                            )
        train_acc = accuracy_score(train_preds, train_labels)
        main_train_accs.append(train_acc) 
        valid_loss, valid_preds, valid_labels, _ = eval_main(
                                                            model = model, 
                                                            iterator = test_loader, 
                                                            criterion = criterion, 
                                                            device = device, 
                                                            args = args
                                                            )
        valid_acc = accuracy_score(valid_preds, valid_labels)
        main_val_accs.append(valid_acc)
        
        # learning rate scheduler
        # Init learing rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)
        scheduler.step(valid_loss)

        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")
        
        if args.adv:
            print("Training discriminators...")
            # Train discriminator until converged
            # evaluate discriminator 
            best_adv_loss, _, _, _ = adv_eval_epoch(
                                                    model = model, 
                                                    discriminators = discriminators, 
                                                    iterator = test_loader, 
                                                    criterion = criterion, 
                                                    device = device, 
                                                    args = args
                                                    )
            best_adv_epoch = -1
            # for k in range(100):
            for k in range(10):
                adv_train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = train_loader, 
                                adv_optimizers = adv_optimizers, 
                                criterion = criterion, 
                                device = device, 
                                args = args
                                )
                adv_valid_loss, adv_valid_preds, _, adv_valid_private_labels = adv_eval_epoch(
                                                        model = model, 
                                                        discriminators = discriminators, 
                                                        iterator = test_loader, 
                                                        criterion = criterion, 
                                                        device = device, 
                                                        args = args
                                                        )
                adv_val_accs = {}
                for j in range(args.n_discriminator):
                    adv_val_accs[j] = accuracy_score(adv_valid_private_labels, adv_valid_preds[j])
                    # print(f"Discriminator {j} Epoch {k+1}/10, Validation Accuracy: {adv_val_accs[j]:.4f}")
        
                # Every 10th epoch, append to disc_val_accs
                if (k+1)//10 == 1:
                    for j in range(args.n_discriminator):
                        if j not in disc_val_accs:
                            disc_val_accs[j] = []
                        disc_val_accs[j].append(adv_val_accs[j])
                        # print(disc_val_accs[j])
                        print(f"Discriminator {j}: Epoch {k+1}/10, Validation Loss: {adv_valid_loss:.4f}, Validation Accuracy: {adv_val_accs[j]:.4f}")

                # # Early stopping for discriminator        
                # if adv_valid_loss < best_adv_loss:
                #         best_adv_loss = adv_valid_loss
                #         best_adv_epoch = k
                #         for j in range(args.n_discriminator):
                #             torch.save(discriminators[j].state_dict(), adv_model_paths[j].format(experiment_type, j))
                # else:
                #     if best_adv_epoch + 5 <= k:
                #         break
        
        if args.use_last == False:
            # Early stopping. This will stop training if the validation loss doesn't improve for 5 epochs.
            # Save the main model if the validation loss is the best we've seen so far.
            # If args.adv is True, then we only save the model if the discriminator accuracy is below 90%
            current_disc_accs = {j: disc_val_accs[j][-1] for j in range(num_discriminators)} if args.adv else None
            if valid_loss < best_loss and i > 0:
                best_acc = valid_acc
                best_loss = valid_loss
                best_epoch = i
                if args.adv:
                    if all(acc < 0.9 for acc in current_disc_accs.values()): # only save if all discriminator accuracies are below 90%
                        print("New best model found and all discriminator accuracies below 90%, saving model...")
                        torch.save(model.state_dict(), main_model_path)
                else:
                    print("New best model found, saving model...")
                    torch.save(model.state_dict(), main_model_path)
            else:
                if best_epoch+5<=i:
                    break
        else:
            print("Saving model from last epoch...")
            torch.save(model.state_dict(), main_model_path)

    # plot main model training and validation loss
    plot_path = f'main_model_accuracy_{experiment_type}'
    if balanced:
        plot_path += '_balanced'
    if masked:
        plot_path += '_masked'
    plot_path += '.png'
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(main_train_accs)+1), main_train_accs, label='Train Accuracy')
    plt.plot(range(1, len(main_val_accs)+1), main_val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Main Model Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    # plt.savefig(plot_path)
    plt.show()

    # Save to correct directory
    main_model_plot_path = os.path.join(output_dir, 'main_model_training_progress.png')
    plt.savefig(main_model_plot_path, dpi=300, bbox_inches='tight')
    print(f"Main model training plot saved to: {main_model_plot_path}")

    disc_plot_path = f'discriminator_accuracy_{experiment_type}'
    if balanced:
        disc_plot_path += '_balanced'
    if masked:
        disc_plot_path += '_masked'
    disc_plot_path += '.png'
    if args.adv:
        # plot discriminator validation accuracy as a subplot of the above
        plt.figure(figsize=(10, 5))
        for j in range(num_discriminators):
            plt.plot(range(1, len(disc_val_accs[j])+1), disc_val_accs[j], label=f'Discriminator {j} Validation Accuracy', color='red')
        plt.xlabel('Epochs (x10)')
        plt.ylabel('Accuracy')
        plt.title('Discriminator Validation Accuracy')
        plt.legend()
        plt.grid()
        # plt.savefig(disc_plot_path)
        plt.show()
        disc_plot_path = os.path.join(output_dir, 'discriminator_training_progress.png')
        plt.savefig(disc_plot_path, dpi=300, bbox_inches='tight')
        print(f"Discriminator training plot saved to: {disc_plot_path}")

    # # Save the plots in the results directory, which is in the parent directory
    # Path("../results").mkdir(parents=True, exist_ok=True)
    # os.replace(plot_path, '../results/'+plot_path)
    # if args.adv:
    #     os.replace(disc_plot_path, '../results/' + disc_plot_path)

    # Load best model for evaluation
    model.load_state_dict(torch.load(main_model_path))
    model.eval()

    return model, main_val_accs, disc_val_accs

######################## OLD CODE FOR RUNNIUNG EXPERIMENTS BELOW, IGNORE ########################

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use_fp16', action='store_true')
#     parser.add_argument('--cuda', type=str)
#     parser.add_argument('--hidden_size', type=int, default = 300)
#     parser.add_argument('--emb_size', type=int, default = 2304)
#     parser.add_argument('--num_classes', type=int, default = 2)
#     parser.add_argument('--adv', action='store_true')
#     parser.add_argument('--adv_level', type=int, default = -1)
#     parser.add_argument('--lr', type=float, default=0.00003)
#     parser.add_argument('--starting_power', type=int)
#     parser.add_argument('--LAMBDA', type=float, default=0.8)
#     parser.add_argument('--n_discriminator', type=int, default = 0)
#     parser.add_argument('--adv_units', type=int, default = 256)
#     parser.add_argument('--ratio', type=float, default=0.8)
#     parser.add_argument('--DL', action='store_true')
#     parser.add_argument('--diff_LAMBDA', type=float, default=1000)
#     parser.add_argument('--data_path', type=str)

#     args = parser.parse_args()

#     # file names
#     experiment_type = "adv_Diverse"
    
#     # path to checkpoints
#     main_model_path = "models\\BERT_model_{}.pt".format(experiment_type)
#     adv_model_path = "models\\discriminator_{}_{}.pt"
    
#     # DataLoader Parameters
#     params = {'batch_size': 512,
#             'shuffle': True,
#             'num_workers': 0}
#     # Device
#     device = torch.device("cuda")

#     data_path = args.data_path
#     # Load data
#     train_data = DeepMojiDataset(args, data_path, "train", ratio=args.ratio, n = 100000)
#     dev_data = DeepMojiDataset(args, data_path, "dev")
#     test_data = DeepMojiDataset(args, data_path, "test")

#     # Data loader
#     training_generator = torch.utils.data.DataLoader(train_data, **params)
#     validation_generator = torch.utils.data.DataLoader(dev_data, **params)
#     test_generator = torch.utils.data.DataLoader(test_data, **params)

#     # Init model
#     model = DeepMojiModel(args)

#     model = model.to(device)

#     # Init discriminators
#     # Number of discriminators
#     n_discriminator = args.n_discriminator

#     discriminators = [Discriminator(args, args.hidden_size, 2) for _ in range(n_discriminator)]
#     discriminators = [dis.to(device) for dis in discriminators]

#     diff_loss = DiffLoss()
#     args.diff_loss = diff_loss

#     # Init optimizers
#     LEARNING_RATE = args.lr
#     optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

#     adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=1e-1*LEARNING_RATE) for dis in discriminators]

#     # Init learing rate scheduler
#     scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

#     # Init criterion
#     criterion = torch.nn.CrossEntropyLoss()
    
    
#     best_loss, valid_preds, valid_labels, _ = eval_main(
#                                                         model = model, 
#                                                         iterator = validation_generator, 
#                                                         criterion = criterion, 
#                                                         device = device, 
#                                                         args = args
#                                                         )

#     best_acc = accuracy_score(valid_labels, valid_preds)
#     best_epoch = 60

#     for i in trange(60):
#         train_epoch(
#                     model = model, 
#                     discriminators = discriminators, 
#                     iterator = training_generator, 
#                     optimizer = optimizer, 
#                     criterion = criterion, 
#                     device = device, 
#                     args = args
#                     )

#         valid_loss, valid_preds, valid_labels, _ = eval_main(
#                                                             model = model, 
#                                                             iterator = validation_generator, 
#                                                             criterion = criterion, 
#                                                             device = device, 
#                                                             args = args
#                                                             )
#         valid_acc = accuracy_score(valid_preds, valid_labels)
#         # learning rate scheduler
#         scheduler.step(valid_loss)

#         # early stopping
#         if valid_loss < best_loss:
#             if i >= 5:
#                 best_acc = valid_acc
#                 best_loss = valid_loss
#                 best_epoch = i
#                 torch.save(model.state_dict(), main_model_path)
#         else:
#             if best_epoch+5<=i:
#                 break

#         # Train discriminator untile converged
#         # evaluate discriminator 
#         best_adv_loss, _, _, _ = adv_eval_epoch(
#                                                 model = model, 
#                                                 discriminators = discriminators, 
#                                                 iterator = validation_generator, 
#                                                 criterion = criterion, 
#                                                 device = device, 
#                                                 args = args
#                                                 )
#         best_adv_epoch = -1
#         for k in range(100):
#             adv_train_epoch(
#                             model = model, 
#                             discriminators = discriminators, 
#                             iterator = training_generator, 
#                             adv_optimizers = adv_optimizers, 
#                             criterion = criterion, 
#                             device = device, 
#                             args = args
#                             )
#             adv_valid_loss, _, _, _ = adv_eval_epoch(
#                                                     model = model, 
#                                                     discriminators = discriminators, 
#                                                     iterator = validation_generator, 
#                                                     criterion = criterion, 
#                                                     device = device, 
#                                                     args = args
#                                                     )
                
#             if adv_valid_loss < best_adv_loss:
#                     best_adv_loss = adv_valid_loss
#                     best_adv_epoch = k
#                     for j in range(args.n_discriminator):
#                         torch.save(discriminators[j].state_dict(), adv_model_path.format(experiment_type, j))
#             else:
#                 if best_adv_epoch + 5 <= k:
#                     break
#         for j in range(args.n_discriminator):
#             discriminators[j].load_state_dict(torch.load(adv_model_path.format(experiment_type, j)))

#     model.load_state_dict(torch.load(main_model_path))
    
#     # Evaluation
#     test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
#     preds = np.array(preds)
#     labels = np.array(labels)
#     p_labels = np.array(p_labels)
    
#     eval_metrices = group_evaluation(preds, labels, p_labels, silence=False)
    
#     print("Overall Accuracy", (eval_metrices["Accuracy_0"]+eval_metrices["Accuracy_1"])/2)