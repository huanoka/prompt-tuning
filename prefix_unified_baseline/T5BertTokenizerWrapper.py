from openprompt.plms.seq2seq import T5TokenizerWrapper


class T5BertTokenizerWrapper(T5TokenizerWrapper):
    def mask_token(self, i):
        return f'extra{id}'

    def mask_token_ids(self, i):
        return self.tokenizer.convert_tokens_to_ids(f'extra{id}')
