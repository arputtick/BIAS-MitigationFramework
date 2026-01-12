# Convert data to torch dataset for classification with gender as label
import torch
from torch.utils.data import Dataset, DataLoader

class BiosDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Create a mapping from profession to numeric labels
        self.mapping = {prof: i for i, prof in enumerate(sorted(self.dataframe['profession'].unique()))}
        self.gender_mapping = {gender: i for i, gender in enumerate(sorted(self.dataframe['gender'].unique()))}
        print('Gender mapping:', self.gender_mapping)
        print('Occupation mapping:', self.mapping)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        bio = self.dataframe.iloc[idx]['hard_text']
        gender = self.dataframe.iloc[idx]['gender']
        label = self.gender_mapping[gender]
        
        # Map professions to numeric labels
        profession = self.dataframe.iloc[idx]['profession']
        occupation_label = self.mapping[profession]


        inputs = self.tokenizer(
            bio,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),   # remove batch dim added by tokenizer
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'occupation_label': torch.tensor(occupation_label, dtype=torch.float)
        }

# ---- DataLoaders with batching ----
def create_dataloaders(train_df, test_df, tokenizer, batch_size=32, max_length=128):
    train_dataset = BiosDataset(train_df, tokenizer, max_length)
    test_dataset = BiosDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader