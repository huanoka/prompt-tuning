import argparse
from torch import nn
from transformers import BertTokenizer
import torch


class BertContextEncoder:

    def __init__(self, tokenizer, model, config):
        self.tokenizer = tokenizer
        self.p1_len = config.p1_length
        self.p2_len = config.p2_length
        self.p3_len = config.p3_length
        self.p4_len = config.p4_length
        self.c_len = config.context_padding_length
        self.q_len = config.question_padding_length
        self.a_len = config.answer_length
        self.model = model
        self.raw_embedding = model.get_input_embedding()
        self.soft_embedding = nn.Embedding(self.p1_len + self.p2_len + self.p3_len + self.p4_len, model.embedding)

    def encode_pairs_bert_batch(self, batch_input):
        for c_id in batch_input['context_ids']:
            c_id.append('[SEP]')
        context_ids, context_mask = self.padding(batch_input['context_ids'], self.c_len)
        question_ids, question_mask = self.padding(batch_input['question_ids'], self.q_len)
        answer_ids, answer_mask = self.padding(batch_input['answer_ids'], self.a_len)
        context_embeddings = self.raw_embedding(context_ids)
        question_embeddings = self.raw_embedding(question_ids)
        answer_embeddings = self.raw_embedding(answer_ids)
        p1_embeddings = self.soft_embedding([i for i in range(self.p1_len)])
        p2_embeddings = self.soft_embedding([i for i in range(self.p1_len, self.p1_len + self.p2_len)])
        p3_embeddings = self.soft_embedding([i for i in range(self.p1_len + self.p2_len, self.p1_len + self.p2_len +
                                                              self.p3_len)])
        p4_embeddings = self.soft_embedding([i for i in range(self.p1_len + self.p2_len + self.p3_len,
                                                              self.p1_len + self.p2_len + self.p3_len + self.p4_len)])
        p1_attentions = torch.ones(p1_embeddings.shape, dtype=torch.long)
        p2_attentions = torch.ones(p2_embeddings.shape, dtype=torch.long)
        p3_attentions = torch.ones(p3_embeddings.shape, dtype=torch.long)
        p4_attentions = torch.ones(p4_embeddings.shape, dtype=torch.long)
        all_embeddings = torch.cat((p1_embeddings, context_embeddings, p2_embeddings, question_embeddings, p3_embeddings,
                                    answer_embeddings, p4_embeddings), dim=1)
        all_attentions = torch.cat((p1_attentions, context_mask, p2_attentions, question_mask, p3_attentions,
                                    answer_mask, p4_attentions), dim=1)

        token_type_ids = torch.zeros_like(all_attentions)
        for b in range(all_attentions.shape[0]):
            for i in range(context_embeddings.shape[1], all_embeddings.shape[1]):
                token_type_ids[b][i] = 1
        return all_embeddings, all_attentions, token_type_ids

    def padding(self, tokens, seq_len, pad_token='[PAD]'):
        # 返回pad后的token, 对应这段的attention_mask
        attention_mask = []
        if len(tokens) > seq_len:
            tokens = tokens[0: seq_len]
            attention_mask = [1 for _ in range(len(seq_len))]
        else:
            attention_mask = [1 for _ in range(len(tokens))]
            while len(tokens) < seq_len:
                tokens.append(pad_token)
                attention_mask.append(0)
        return tokens, attention_mask


def test():
    context = 'All biomass goes through at least some of these steps: it needs to be grown  collected  dried  fermented  ' \
              'distilled  and burned. All of these steps require resources and an infrastructure. The total amount of energy' \
              ' input into the process compared to the energy released by burning the resulting ethanol fuel is known as the' \
              ' energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National ' \
              'Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel ' \
              'energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ' \
              'ethanol produced in Brazil is more favorable  with one unit of fossil-fuel energy required to create 8 from ' \
              'the ethanol. Energy balance estimates are not easily produced  thus numerous such reports have been generated' \
              ' that are contradictory. For instance  a separate survey reports that production of ethanol from sugarcane  ' \
              'which requires a tropical climate to grow productively  returns from 8 to 9 units of energy for each unit ' \
              'expended  as compared to corn  which only returns about 1.34 units of fuel energy for each unit of energy ' \
              'expended. A 2006 University of California Berkeley study  after analyzing six separate studies  concluded ' \
              'that producing ethanol from corn uses much less petroleum than producing gasoline.'

    question = 'does ethanol take more energy make that produces'
    ans = 'FALSE'
    config = argparse.Namespace(
        p1_length=5,
        p2_length=10,
        p3_length=5,
        context_len=450,
        question_len=39,
        answer_len=1
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ids, types, masks = encode_text_bert(tokenizer, context, question, ans, config)
    print(ids)
    print(types)
    print(masks)


test()
