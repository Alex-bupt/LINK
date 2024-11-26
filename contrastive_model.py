import json
import itertools
import math
import os
import random
import argparse
import sys
import time
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import nn
from torch_geometric.nn import RGCNConv
from collections import defaultdict
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from transformers import AutoTokenizer, AutoModel
from dataset_pre2 import CRSDataset, CRSDataCollator

from dataset_dbpedia import DBpedia
from model_gpt2 import PromptGPT2forCRS
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from Qformer.models.blip2_models.fusion_qformer import EntityQformer
import torch.distributed as dist
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import wandb
from transformers import AdamW, get_linear_schedule_with_warmup


class TripletDataset(Dataset):
    def __init__(self):
        self.data = self.load_dataset()
        self.entity_descriptions = self.load_entity_descriptions()

    def load_dataset(self):
        unique_entities = set()
        for filename in ['train_data_processed.jsonl', 'valid_data_processed.jsonl', 'test_data_processed.jsonl']:
            with open(f'/root/autodl-tmp/UniCRS-main/src/data/redial/{filename}', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    entities = data.get("entity", [])
                    unique_entities.update(entities)

        return unique_entities

    def load_entity_descriptions(self):
        # 读取 movies_with_ids.json 文件
        with open('/root/autodl-tmp/UniCRS-main/src/data/redial/movies_with_ids.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)

        # 初始化一个长度为 unique_entities 中最大值加一的列表
        max_id = max(self.data)
        entity_descriptions = [""] * (max_id + 1)

        # 将描述放在对应的索引位置上
        for entity_id, description in movies_data.items():
            int_id = int(entity_id)
            if int_id <= max_id:
                entity_descriptions[int_id] = description

        return entity_descriptions


class ContrastiveLearningModel(nn.Module):
    def __init__(
            self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
            n_entity, num_relations, num_bases, edge_index, edge_type,
            n_prefix_rec=None, n_prefix_conv=None
    ):
        super(ContrastiveLearningModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.Qformer = EntityQformer()

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, context_str=None, device=None):
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_size, entity_len = entity_ids.shape[:2]
        entity_embeds = self.get_entity_embeds()
        entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        loss = self.Qformer(entity_embeds, context_str)

        return loss

        # Initialize loss tensor on the specified device
        # loss = torch.tensor(0.0, device=device, requires_grad=True)


#         #
#         # # Get embeddings and move them to the specified device
#         # embeds = self.get_embedding().to(device)
#         #
#         # batch_size, entity_num = entity_ids.shape
#         #
#         # for b in range(batch_size):
#         #     # Collect valid entity IDs that are not pad_id
#         #     valid_mask = entity_ids[b] != pad_id
#         #     valid_entity_ids = entity_ids[b][valid_mask]
#         #
#         #     # If there are no valid entity IDs, skip this batch
#         #     if valid_entity_ids.size(0) == 0:
#         #         continue
#         #
#         #     # If there is only one valid entity ID, continue
#         #     if valid_entity_ids.size(0) == 1:
#         #         continue
#         #
#         #     entity_embeds = embeds[valid_entity_ids].view(-1, 1, self.hidden_size)
#         #     descriptions = [entity_descriptions[eid.item()] for eid in valid_entity_ids]
#         #
#         #     # Calculate loss using Qformer and accumulate
#         #     batch_loss = self.Qformer(descriptions, entity_embeds)
#         #
#         #     # Ensure that loss is not accumulated in-place on the leaf variable
#         #     loss = loss + batch_loss
#         #
#         # return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument("--fp16", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


def evaluate(dataloader, device, text_encoder, model):
    model.eval()
    eval_loss = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            loss = model(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                context_str=batch['context_str'],
                output_entity=True,
                device=device
            )
        eval_loss.append(float(loss))
    return np.mean(eval_loss)


def main(rank, world_size):
    learning_rate = 0.001
    model_path = 'best_model.pth'

    args = parse_args()
    config = vars(args)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG')
    if rank == 0:
        logger.add(f'log/{local_time}.log', level='DEBUG')
    logger.info(config)

    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.output_dir is not None and rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length, debug=args.debug,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    # 初始化模型
    model = ContrastiveLearningModel(768, text_encoder.config.hidden_size, 12,
                                     12, 2,
                                     n_entity=kg['num_entities'], num_relations=kg['num_relations'],
                                     num_bases=args.num_bases,
                                     edge_index=kg['edge_index'], edge_type=kg['edge_type']).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 每个epoch后学习率衰减为原来的0.5倍

    # 训练模型
    best_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        train_loss = []
        model.train()

        # train
        for step, batch in enumerate(tqdm(train_dataloader, disable=(rank != 0))):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            loss = model(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                context_str=batch['context_str'],
                output_entity=True,
                device=device
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = sum(train_loss) / len(train_loss)

        print(f'\epoch {epoch} train loss {avg_loss:.4f}')

        if rank == 0:
            logger.info(f'epoch {epoch} train loss {avg_loss:.4f}')

        valid_loss = evaluate(valid_dataloader, device, text_encoder, model)

        print(f'epoch {epoch} valid loss {valid_loss:.4f}')

        # 保存最佳模型
        if rank == 0 and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with loss: {best_loss:.4f}')

        scheduler.step()  # 每个epoch结束后进行学习率衰减

    dist.destroy_process_group()


def main_worker(rank, world_size):
    main(rank, world_size)


if __name__ == "__main__":
    args = parse_args()
    world_size = 1  # Number of GPUs or processes
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

    # args = parse_args()
    # config = vars(args)
    #
    # # Initialize the accelerator. We will let the accelerator handle device placement for us.
    # accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    # device = accelerator.device
    #
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
    #
    # # Make one log on every process with the configuration for debugging.
    # local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # logger.remove()
    # logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    # logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    # logger.info(config)
    # logger.info(accelerator.state)
    #
    # if accelerator.is_local_main_process:
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     transformers.utils.logging.set_verbosity_error()
    #
    # # wandb
    # if args.use_wandb:
    #     name = args.name if args.name else local_time
    #     name += '_' + str(accelerator.process_index)
    #
    #     if args.log_all:
    #         group = args.name if args.name else 'DDP_' + local_time
    #         run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
    #     else:
    #         if accelerator.is_local_main_process:
    #             run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
    #         else:
    #             run = None
    # else:
    #     run = None
    #
    # # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #
    # if args.output_dir is not None:
    #     os.makedirs(args.output_dir, exist_ok=True)
    #
    # kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    #
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    # model = PromptGPT2forCRS.from_pretrained(args.model)
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model = model.to(device)
    #
    # text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    # text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    # text_encoder = AutoModel.from_pretrained(args.text_encoder)
    # text_encoder.resize_token_embeddings(len(text_tokenizer))
    # text_encoder = text_encoder.to(device)
    #
    # # Initialize and load ContrastiveLearningModel
    # contrastive_model = ContrastiveLearningModel(
    #     hidden_size=model.config.hidden_size,
    #     token_hidden_size=text_encoder.config.hidden_size,
    #     n_head=model.config.num_attention_heads,
    #     n_layer=model.config.num_hidden_layers,
    #     n_block=2,
    #     n_entity=kg['num_entities'],
    #     num_relations=kg['num_relations'],
    #     num_bases=args.num_bases,
    #     edge_index=kg['edge_index'],
    #     edge_type=kg['edge_type'],
    # )
    # contrastive_model = contrastive_model.to(device)
    #
    # # optim & amp
    # modules = [contrastive_model]
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for model in modules for n, p in model.named_parameters()
    #                    if not any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for model in modules for n, p in model.named_parameters()
    #                    if any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # # data
    # train_dataset = CRSDataset(
    #     dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
    #     prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    #     entity_max_length=args.entity_max_length
    # )
    # valid_dataset = CRSDataset(
    #     dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
    #     prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    #     entity_max_length=args.entity_max_length
    # )
    # test_dataset = CRSDataset(
    #     dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
    #     prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    #     entity_max_length=args.entity_max_length
    # )
    # data_collator = CRSDataCollator(
    #     tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
    #     max_length=args.max_length, entity_max_length=args.entity_max_length,
    #     use_amp=accelerator.use_fp16, debug=args.debug,
    #     prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.per_device_train_batch_size,
    #     collate_fn=data_collator,
    #     shuffle=True
    # )
    # valid_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size=args.per_device_eval_batch_size,
    #     collate_fn=data_collator,
    #     shuffle=False
    # )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=args.per_device_eval_batch_size,
    #     collate_fn=data_collator,
    #     shuffle=False
    # )
    #
    # # step, epoch, batch size
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # else:
    #     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # completed_steps = 0
    # # lr_scheduler
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    # lr_scheduler = accelerator.prepare(lr_scheduler)
    # # training info
    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    #
    # # save model with best metric
    # metric, mode = 'loss', -1
    # assert mode in (-1, 1)
    # if mode == 1:
    #     best_metric = 0
    # else:
    #     best_metric = float('inf')
    # best_metric_dir = os.path.join(args.output_dir, 'best')
    # os.makedirs(best_metric_dir, exist_ok=True)
    #
    #
    # def evaluate(dataloader):
    #     model.eval()
    #     eval_loss = []
    #     for step, batch in enumerate(dataloader):
    #         with torch.no_grad():
    #             token_embeds = text_encoder(**batch['prompt']).last_hidden_state
    #             loss = contrastive_model(
    #                 entity_ids=batch['entity'],
    #                 token_embeds=token_embeds,
    #                 context_str=batch['context_str'],
    #                 output_entity=True
    #             )
    #         eval_loss.append(float(loss))
    #     return np.mean(eval_loss)
    #
    #
    # for epoch in range(args.num_train_epochs):
    #     train_loss = []
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         with torch.no_grad():
    #             token_embeds = text_encoder(**batch['prompt']).last_hidden_state
    #         loss = contrastive_model(
    #             entity_ids=batch['entity'],
    #             token_embeds=token_embeds,
    #             context_str=batch['context_str'],
    #             output_entity=True
    #         )
    #
    #         accelerator.backward(loss)
    #         train_loss.append(float(loss))
    #
    #         if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
    #             if args.max_grad_norm is not None:
    #                 accelerator.clip_grad_norm_(contrastive_model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #
    #             progress_bar.update(1)
    #             completed_steps += 1
    #             if run:
    #                 run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})
    #
    #         if completed_steps >= args.max_train_steps:
    #             break
    #
    #     train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
    #     logger.info(f'epoch {epoch} train loss {train_loss}')
    #
    #     valid_loss = evaluate(valid_dataloader)
    #     logger.info(f'epoch {epoch} valid loss {valid_loss}')
    #
    #     if (mode == 1 and valid_loss > best_metric) or (mode == -1 and valid_loss < best_metric):
    #         best_metric = valid_loss
    #         logger.info(f"Saving model with best {metric}: {best_metric}")
    #         accelerator.save_state(best_metric_dir)
    #
    #     if run:
    #         run.log({'train_loss': train_loss, 'valid_loss': valid_loss})
    #
    #     del train_loss, valid_loss, batch
    #
    # # Final evaluation on test dataset
    # test_loss = evaluate(test_dataloader)
    # logger.info(f'test loss: {test_loss}')
    # if run:
    #     run.log({'test_loss': test_loss})
    #
    # # Close wandb run
    # if run:
    #     run.finish()
