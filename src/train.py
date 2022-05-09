import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import time
from tqdm.auto import tqdm
import numpy as np

import torch
from torch.optim import AdamW
import datasets
import transformers
from transformers import get_scheduler, SchedulerType, set_seed
from datasets import load_metric
from accelerate import Accelerator, DistributedDataParallelKwargs
from model import XQBert, XQSBert
from dataset import load_concat_train_data, load_concat_dev_data, load_paired_train_data, load_paired_dev_data

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="concat"
    )
    parser.add_argument(
        "--backbone", type=str, default="bert"
    )
    parser.add_argument(
        "--embedding_method", type=str, default="cls_with_pooler"
    )
    parser.add_argument(
        "--train_data_path", type=str, default=None
    )
    parser.add_argument(
        "--dev_data_path", type=str, default=None
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--per_device_dev_batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--llrd_factor", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_ratio", type=float, default=0.0)
    parser.add_argument("--classifier_dropout_rate", type=float, default=None)
    parser.add_argument("--reinit_layers", type=int, default=0)
    parser.add_argument("--lr_scale_factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()
    return args

def get_opt_grouped_params_llrd(model, top_lr, weight_decay, lr_decay, classifier_scale_factor):
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = top_lr
    head_lr = top_lr + 1e-6
    classifier_lr = top_lr * classifier_scale_factor
    lr = init_lr
    
    # === Classifier ====================================================== 
   
    classifier_names = ["classifier.dense.weight", "classifier.out_proj.weight", "classifier.weight"]
    classifer_parameters = {
        "params": [p for n,p in named_parameters if any(nd in n for nd in classifier_names)],
        "weight_decay": weight_decay,
        "lr": classifier_lr,
    }
    opt_parameters.append(classifer_parameters)

    classifier_names = ["classifier.dense.bias", "classifier.out_proj.bias", "classifier.bias"]
    classifer_parameters = {
        "params": [p for n,p in named_parameters if any(nd in n for nd in classifier_names)],
        "weight_decay": 0.0,
        "lr": classifier_lr,
    }
    opt_parameters.append(classifer_parameters)

    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
    opt_parameters.append(head_params)
                
    # === All Hidden layers ==========================================================
   
    for layer in range(model.config.num_hidden_layers, -1, -1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)       
        
        lr *= lr_decay
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay} 
    opt_parameters.append(embed_params)      

    return opt_parameters

def train():
    args = parse_args()
    assert(args.log_dir != None)
    use_dev = True
    if args.dev_data_path is None:
        use_dev = False

    if args.model_type == "paired":
        kwargs_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    else:
        accelerator = Accelerator()

    logging.basicConfig(
        filename=os.path.join(args.log_dir, time.strftime("%m%d%H%M%S") + '.log'),
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        if args.seed < 0:
            # args.seed = np.random.randint(1024)
            args.seed = np.random.randint(np.iinfo(np.uint32).max)
        set_seed(args.seed)

    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        if args.model_type == 'concat':
            train_dataloader = load_concat_train_data(args.backbone, args.train_data_path, args.per_device_train_batch_size, padding, args.max_length)
            if use_dev:
                dev_dataloader = load_concat_dev_data(args.backbone, args.dev_data_path, args.per_device_dev_batch_size, padding, args.max_length)
        elif args.model_type == 'paired':
            train_dataloader = load_paired_train_data(args.backbone, args.train_data_path, args.per_device_train_batch_size, padding, args.max_length)
            if use_dev:
                dev_dataloader = load_paired_dev_data(args.backbone, args.dev_data_path, args.per_device_dev_batch_size, padding, args.max_length)
        else:
            raise ValueError('Unknown model type {}'.format(args.model_type))
    
    if args.model_type == 'concat':
        model = XQBert(backbone=args.backbone, num_labels=2, dropout=args.classifier_dropout_rate, embedding_method=args.embedding_method, reinit_layers=args.reinit_layers)
    else:
        model = XQSBert(backbone=args.backbone, num_labels=2, dropout=args.classifier_dropout_rate, embedding_method=args.embedding_method)

    optimizer_grouped_parameters = get_opt_grouped_params_llrd(model, args.learning_rate, args.weight_decay, args.llrd_factor, args.lr_scale_factor)
    optimizer = AdamW(optimizer_grouped_parameters)
    # if args.backbone == 'bert':
    #     optimizer = AdamW(optimizer_grouped_parameters)
    # elif args.backbone == 'roberta':
    #     optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98))
    # else:
    #     raise ValueError("Unsupported backbone model {}".format(args.backbone))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.num_warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )

    if use_dev:
        model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    metric = load_metric("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Hyper Parameters *****")
    logger.info(f"  Model Type = {args.model_type}")
    logger.info(f"  Backbone = {args.backbone}")
    logger.info(f"  Embedding Method = {args.embedding_method}")
    logger.info(f"  Re-init Top {args.reinit_layers} layers")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Max Padding Length = {args.max_length}")
    logger.info(f"  Pad to max length = {args.pad_to_max_length}")
    logger.info(f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Learning Rate = {args.learning_rate}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  Layer-wise Learning Rate Decay = {args.llrd_factor}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Num Warmup Steps = {int(args.num_warmup_ratio * num_training_steps)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    logger.info(f"  Use dev dataset = {use_dev}")
    logger.info(f"  Classifier Dropout Rate = {args.classifier_dropout_rate}")
    logger.info(f"  Learning Rate Scale Factor (classifier : backbone) = {args.lr_scale_factor}")
    logger.info(f"  Seed = {args.seed}")
    logger.info("***** Start Training *****")

    acc = 0.0
    progress_bar = tqdm(range(num_training_steps // accelerator.num_processes), disable=not accelerator.is_local_main_process)
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        if use_dev:
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(dev_dataloader):
                    outputs = model(batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch['labels'])
                    )
                eval_metric = metric.compute()
                logger.info(f"Epoch {epoch}: {eval_metric}")
                logger.info(f"Epoch {epoch}: loss = {total_loss}")
                if eval_metric['accuracy'] > acc:
                    acc = eval_metric['accuracy']
                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        if args.model_type == 'concat':
                            output_name = 'XQBert-' + time.strftime("%m%d-%H%M%S") + '.pth'
                        else:
                            output_name = 'XQSBert-' + time.strftime("%m%d-%H%M%S") + '.pth'
                        output_path = os.path.join(args.output_dir, output_name)
                        torch.save({
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict() 
                        }, output_path)
                        logger.info(f"Model save to {output_path}")
        else:
            logger.info(f"Epoch {epoch} Done")
            logger.info(f"Epoch {epoch}: loss = {total_loss}")
            # if epoch == args.num_train_epochs - 1:
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                if args.model_type == 'concat':
                    output_name = 'XQBert-' + time.strftime("%m%d-%H%M%S") + '.pth'
                else:
                    output_name = 'XQSBert-' + time.strftime("%m%d-%H%M%S") + '.pth'
                output_path = os.path.join(args.output_dir, output_name)
                torch.save({
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() 
                }, output_path)
                logger.info(f"Model save to {output_path}")

if __name__ == '__main__':     
    train()