""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from config import args
from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
# from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification # chinese version
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.compute_metrics import metrics
from processors import output_modes
from processors import processors
from processors import convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
            

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #Log metrics
                if args.local_rank == -1:
                    evaluate(args, model, tokenizer)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
            pbar(step, {'loss': loss.item()})
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return results

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.max_seq_length,
                                            output_mode = output_mode)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

def predict(args, model, tokenizer, prefix=""):

    predict_task_names = (args.task_name,)
    predict_outputs_dirs = (args.output_dir,)

    results = {}
    for predict_task, predict_output_dir in zip(predict_task_names, predict_outputs_dirs):
        predict_dataset = load_and_cache_examples(args, predict_task, tokenizer, data_type='predict')
        if not os.path.exists(predict_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(predict_output_dir)
        per_gpu_predict_batch_size = args.per_gpu_eval_batch_size
        args.predict_batch_size = per_gpu_predict_batch_size * max(1, args.n_gpu)
        predict_sampler = SequentialSampler(predict_dataset)# if args.local_rank == -1 else DistributedSampler(predict_dataset)
        predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.predict_batch_size,
                                     collate_fn=collate_fn)

        # predict!
        logger.info("***** Running predict {} *****".format(prefix))
        logger.info("  Num examples = %d", len(predict_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        predict_loss = 0.0
        nb_predict_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(predict_dataloader), desc="predict")
        for step, batch in enumerate(predict_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                tmp_predict_loss, logits = outputs[:2]
                predict_loss += tmp_predict_loss.mean().item()
            nb_predict_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        predict_loss = predict_loss / nb_predict_steps
        preds = np.argmax(preds, axis=1)
        logger.info('预测结果是{}'.format(preds))
        preds_out_path = os.path.join(args.output_dir, 'preds_out.txt')
        with open(preds_out_path, 'w', encoding='utf-8') as f:
            for i in range(len(preds)):
                f.write('预测%s-真实%s\n' % (preds[i], out_label_ids[i]))
        result = metrics(predict_task, preds, out_label_ids)
        logger.info('计算矩阵是{}'.format(result))
        results.update(result)
        logger.info("***** predict results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return results

def main():

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    type_task = args.model_type + '_' + '{}'.format(args.task_name)
    if not os.path.exists(os.path.join(args.output_dir, type_task)):
        os.mkdir(os.psth.join(args.output_dir, type_task))
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    seed_everything(args.seed)
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AlbertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                 )
    model =AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,                                                            config=config)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    args.do_train = True
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    args.do_eval = True
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file,
                                                      do_lower_case=args.do_lower_case,
                                                      )
        checkpoints = [(0,args.output_dir)]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]),checkpoint) for checkpoint in checkpoints if checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints,key =lambda x:x[0])
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _,checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model =AlbertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key,value in results:
                writer.write("%s = %s\n" % (key, str(value)))

    args.do_predict = True
    predict_results = []
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file,
                                                      do_lower_case=args.do_lower_case,
                                                      )
        checkpoints = [(0, args.output_dir)]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]), checkpoint) for checkpoint in checkpoints if
                           checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints, key=lambda x: x[0])
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _, checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = AlbertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = predict(args, model, tokenizer, prefix=prefix)
            predict_results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key, value in predict_results:
                writer.write("%s = %s\n" % (key, str(value)))

if __name__ == "__main__":
    main()
