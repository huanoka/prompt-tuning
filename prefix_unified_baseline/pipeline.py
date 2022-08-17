import argparse
import json
import time

import openprompt.utils.crossfit_metrics
import torch
from openprompt.prompts import PrefixTuningTemplate, GenerationVerbalizer, MixedTemplate
from openprompt import PromptForGeneration
from openprompt.utils.metrics import generation_metric
from transformers import BertTokenizer, MT5ForConditionalGeneration
from T5BertTokenizerWrapper import T5BertTokenizerWrapper
from openprompt import PromptDataLoader
from utils import load_dataset
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from tqdm import tqdm
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate


args = argparse.Namespace(
    max_seq_length=512,
    decoder_max_length=256,
    epoch=100,
    tune_plm=False,
    plm_lr=5e-4,
    prompt_lr=0.3,
    batch_size=16,
    device='cuda:3',
    warmup_step=1000,
    test_res_path='../results/gen_result.txt'
)
generation_args = argparse.Namespace(
    max_length=args.decoder_max_length
)
def evaluate(val_loader, model, device, gen_config=None):
    model.eval()
    ground_truths = []
    predictions = []
    contexts = []
    questions = []
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(val_loader), total=len(val_loader), desc='validating...'):
            inputs = inputs.to(device)
            _, output_sentence = model.generate(inputs, **gen_config)
            ground_truth = inputs['tgt_text']
            contexts.extend(inputs['text_a'])
            questions.extend(inputs['text_b'])
            output_sentence = [o_s.strip() for o_s in output_sentence]
            ground_truth = [g_t.strip() for g_t in ground_truth]
            ground_truths.extend(ground_truth)
            predictions.extend(output_sentence)
        acc = crossfit_evaluate(predictions, ground_truths, metric='QA-F1')
    return acc, contexts, questions, ground_truths, predictions


def prefix_pipeline(config):
    tokenizer = BertTokenizer.from_pretrained('uer/t5-v1_1-base-chinese-cluecorpussmall')
    plm = MT5ForConditionalGeneration.from_pretrained('uer/t5-v1_1-base-chinese-cluecorpussmall')
    tokenizer.eos_token = 'extra10'
    tokenizer_wrapper = T5BertTokenizerWrapper(max_seq_length=512, tokenizer=tokenizer, truncate_method='tail',
                                               decoder_max_length=256, predict_eos_token=True)
    mix_template = MixedTemplate(model=plm, tokenizer=tokenizer,
                                 text='{"soft", "duplicate": 20} Context: {"placeholder": "text_a", "shortenable": True}'
                                      + '{"soft", "duplicate": 20} Question: {"placeholder": "text_b"} Answer: {"mask"}')

    prefix_template = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, num_token=100,
                                           text='Context: {"placeholder": "text_a", "shortenable": True} Question: ' +
                                           '{"placeholder": "text_b"} Answer: {"mask"}')

    dataset = load_dataset()
    train_loader = PromptDataLoader(dataset=dataset['train'], template=prefix_template, tokenizer=tokenizer,
                                    tokenizer_wrapper=tokenizer_wrapper, max_seq_length=config.max_seq_length,
                                    decoder_max_length=config.decoder_max_length,
                                    batch_size=config.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                    truncate_method='tail')
    dev_loader = PromptDataLoader(dataset=dataset['dev'], template=prefix_template, tokenizer=tokenizer,
                                  tokenizer_wrapper=tokenizer_wrapper, max_seq_length=config.max_seq_length,
                                  decoder_max_length=config.decoder_max_length,
                                  batch_size=config.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                  truncate_method='tail')
    test_loader = PromptDataLoader(dataset=dataset['test'], template=prefix_template, tokenizer=tokenizer,
                                   tokenizer_wrapper=tokenizer_wrapper, max_seq_length=config.max_seq_length,
                                   decoder_max_length=config.decoder_max_length,
                                   batch_size=config.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                   truncate_method='tail')
    model = PromptForGeneration(plm, mix_template,
                                freeze_plm=(not config.tune_plm), plm_eval_mode=False, tokenizer=tokenizer)
    device = torch.device(config.device)
    model = model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    if config.tune_plm:
        optimizer_grouped_paramters = [
            {'params': [p for n, p in model.plm.named_parameters() if not any([nd in n for nd in no_decay])], 'weight_decay':0.01},
            {'params': [p for n, p in model.plm.named_parameters() if any([nd in n for nd in no_decay])], 'weight_decay':0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_paramters, lr=config.plm_lr)
        scheduler1 = get_constant_schedule_with_warmup(optimizer1, num_warmup_steps=config.warmup_step)
    else:
        optimizer1 = None
        scheduler1 = None
    optimizer_grouped_paramters2 = [
        {'params': [p for n, p in model.template.named_parameters() if 'raw_embedding' not in n]}
    ]
    tot_step = len(train_loader) * config.epoch
    optimizer2 = AdamW(optimizer_grouped_paramters2, lr=config.prompt_lr)
    # scheduler2 = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer2, config.warmup_step,
    #                                                                 num_training_steps=tot_step, num_cycles=tot_step//10)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, config.warmup_step, tot_step)
    optimizers = [optimizer1, optimizer2]
    schedulers = [scheduler1, scheduler2]
    best_acc = 0.0
    for epoch in range(config.epoch):
        model.train()
        tot_loss = 0.0
        for step, inputs in tqdm(enumerate(train_loader), tot_step=len(train_loader), desc='training...'):
            inputs = inputs.to(device)
            loss = model(inputs)
            loss.backward()
            tot_loss += loss.item()
            for optimizer, scheduler in zip(optimizers, schedulers):
                if optimizer is not None:
                    optimizer.step()
                    scheduler.step()
        acc, _, _, _, _ = evaluate(val_loader=dev_loader, model=model, device=device, gen_config=generation_args)
        improved = ''
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), '../ckpt/prompt_model.pkl')
            improved = '*'
        print('epoch:{}, train_loss{:.6f}, val_acc:{:.6f} {}'.format(epoch, tot_loss/(len(train_loader)), acc, improved))
    acc, contexts, questions, ground_truths, predictions = evaluate(test_loader, model, device, generation_args)
    out_file = open(args.test_res_path, 'w', encoding='utf-8')
    out_file.write(f'best_acc: {best_acc}, test_acc: {acc}\n')
    for con, qu, gt, pr in zip(contexts, questions, ground_truths, predictions):
        out_file.write(f'context: {con}\tqu: {qu}\tge: {ge}\tpr:{pr}\n')


prefix_pipeline(config)
