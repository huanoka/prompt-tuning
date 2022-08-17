from openprompt.data_utils import InputExample


def load_dataset(includes=None):
    if includes is None:
        includes = (
            'boolq',
            'commonsense',
            'drop',
            'MCTest_160',
            'MCTest_500',
            'squad1.1'
        )
    dataset = {'train': [], 'dev': [], 'test': []}
    global_ids = {'train': 0, 'dev':0, 'test': 0}
    for ds_name in includes:
        for split in ['train', 'dev', 'test']:
            with open('../datasets/' + ds_name + '/' + split + '_unified.jsonl', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    input_example = InputExample(guid=global_ids[split], text_a=line['context'], text_b=line['question'],
                                                 tgt_text=line['answer'])
                    global_ids[split] += 1
                    dataset[split].append(input_example)
    print('dataset加载完毕, train {}, dev {}, test {}'.format(len(dataset['train'], len(dataset['dev'], len(dataset['test'])))))
    return dataset
