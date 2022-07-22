import argparse

use_config = argparse.Namespace(
    p1_length=5,
    p2_length=10,
    p3_length=5,
    p4_length=5,
    context_len=450,
    question_len=39,
    answer_len=1,
    model_name='bert-base-uncased',
    embedding_size=768
)