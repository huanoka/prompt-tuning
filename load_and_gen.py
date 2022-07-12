import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5PreTrainedModel, T5Tokenizer


class QADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return {
            'context': self.df.iloc[item]['context'],
            'question': self.df.iloc[item]['question'],
            'answer': self.df.iloc[item]['answer']
        }


def load_dataset(path, config):
    df = pd.read_csv(path, encoding='utf-8')
    dataset = QADataset(df)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=config.drop_last)
    return loader


def load_model(config):
    t5_model = T5PreTrainedModel.from_pretrained(config.T5_model_type)
    tokenizer = T5Tokenizer.from_pretrained(config.T5_model_type)
    for i in range(config.p_length):
        to_add = '[prompt' + str(i) + ']'
        tokenizer.add_special_token(to_add)
    t5_model.resize_token_embeddings(config.p_length)
    return t5_model, tokenizer









