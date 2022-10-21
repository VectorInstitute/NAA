from torch.utils.data import DataLoader

import DataLoader.MARCOData as MARCOData
import DataLoader.SQuADData as SQuADData
import DataLoader.UbuntuData as UbuntuData
import DataLoader.GeneralData as GeneralData
from DataLoader.dataset_utils import *
from utils import *


def get_dataloaders(args, tokenizer):
    """
    Prepare the dataloaders for training and evaluation.

    args should be a namespace that gives max_len, train_batch_size, valid_batch_size, distributed arguments.
    """

    print("Building datasets")
    if args.dataset_type == "marco":
        train_dataset = MARCOData.MARCODataset(args, tokenizer, args.train_dataset_path)
        valid_dataset = MARCOData.MARCODataset(
            args, tokenizer, args.valid_dataset_path, limit=1000
        )
    elif args.dataset_type == "squad":
        train_dataset = SQuADData.SQuADDataset(args, tokenizer, args.train_dataset_path)
        valid_dataset = SQuADData.SQuADDataset(args, tokenizer, args.valid_dataset_path)
    elif args.dataset_type == "ubuntu" and args.second_loss == "mc":
        train_dataset = UbuntuData.UbuntuMCDataset(
            args, tokenizer, args.train_dataset_path
        )
        valid_dataset = UbuntuData.UbuntuMCDataset(
            args, tokenizer, args.valid_dataset_path
        )
    elif args.dataset_type == "ubuntu" and args.second_loss == "bc":
        train_dataset = UbuntuData.UbuntuBCDataset(
            args, tokenizer, args.train_dataset_path
        )
        valid_dataset = UbuntuData.UbuntuBCDataset(
            args, tokenizer, args.valid_dataset_path
        )
    elif args.dataset_type == "general":
        train_dataset = GeneralData.GeneralMCDataset(
            args, tokenizer, args.train_dataset_path
        )
        valid_dataset = GeneralData.GeneralMCDataset(
            args, tokenizer, args.valid_dataset_path
        )
    else:
        raise ValueError(
            "args.dataset {} and/or args.second_loss {} are incompatible".format(
                args.dataset, args.second_loss
            )
        )

    print("Building dataloaders")
    if args.second_loss == "mc":
        train_sampler = StatefulSampler(train_dataset, True)
        valid_sampler = StatefulSampler(valid_dataset, False)
        if args.distributed:
            train_sampler = DistributedSamplerWrapper(
                train_sampler, num_replicas=args.num_gpus, rank=args.local_rank
            )
            valid_sampler = DistributedSamplerWrapper(
                valid_sampler,
                num_replicas=args.num_gpus,
                rank=args.local_rank,
                shuffle=False,
            )

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=padding_collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=args.valid_batch_size,
            collate_fn=padding_collate_fn,
        )
    else:
        if args.distributed:
            train_sampler = DistributedBCBatchSampler(
                train_dataset,
                num_replicas=args.num_replicas,
                rank=args.local_rank,
                drop_last=True,
                shuffle=False,
                batch_size=args.train_batch_size,
            )
            train_sampler = DistributedBCBatchSampler(
                valid_dataset,
                num_replicas=args.num_replicas,
                rank=args.local_rank,
                drop_last=True,
                shuffle=False,
                batch_size=args.valid_batch_size,
            )
        else:
            train_sampler = BCBatchSampler(
                train_dataset, args.train_batch_size, shuffle=True, drop_last=True
            )
            valid_sampler = BCBatchSampler(
                valid_dataset, args.valid_batch_size, shuffle=False, drop_last=True
            )

        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=padding_bc_collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_sampler=valid_sampler, collate_fn=padding_bc_collate_fn
        )

    print("Train Dataset Length: {:d}".format(len(train_dataset)))
    print("Valid Dataset Length: {:d}".format(len(valid_dataset)))

    return train_loader, valid_loader, train_sampler, valid_sampler


def get_validation_dataloader(args, tokenizer):
    """
    Prepare a dataloader for evaluation. Assumes this is just for the evaluation step.

    args should be a namespace that gives max_len, valid_batch_size arguments.
    """

    print("Building datasets")
    if args.dataset_type == "marco":
        valid_dataset = MARCOData.MARCODataset(args, tokenizer, args.valid_dataset_path)
    elif args.dataset_type == "squad":
        valid_dataset = SQuADData.SQuADDataset(args, tokenizer, args.valid_dataset_path)
    elif args.dataset_type == "ubuntu":
        valid_dataset = UbuntuData.UbuntuMCDataset(
            args, tokenizer, args.valid_dataset_path
        )
    elif args.dataset_type == "general":
        valid_dataset = GeneralData.GeneralMCDataset(
            args, tokenizer, args.valid_dataset_path
        )
    else:
        raise ValueError(
            "args.dataset {} and/or args.second_loss {} are incompatible".format(
                args.dataset, args.second_loss
            )
        )

    print("Building dataloaders")
    valid_sampler = StatefulSampler(valid_dataset, False)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=args.valid_batch_size,
        collate_fn=padding_collate_fn,
    )

    print("Valid Dataset Length: {:d}".format(len(valid_dataset)))

    return valid_loader
