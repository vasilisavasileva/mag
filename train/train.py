# %%
import math
import os

import torch
import datasets
import transformers

from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from catalyst import dl, utils, callbacks

from model import RecModel


SEED = 69

# %%
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# %%

model = RecModel(in_dim=798, out_dim=797, n_head=6)
# model.load_state_dict(torch.load("./checkpoints/model.last.pth"))

# %%
train_dataset = datasets.load_from_disk("train_users_dataset")
test_dataset = datasets.load_from_disk("test_users_dataset")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# %%
BATCH_SIZE = 1024

# %%
loaders = {
    "train": DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
    "valid": DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
}

# %%

EPOCHS = 100

optimizer = optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=15, num_training_steps=EPOCHS)

# %%
runner = dl.SupervisedRunner(
    model=model,
    input_key="data",
    output_key="logits",
    target_key="target",
    loss_key="loss",
    engine=dl.GPUEngine()
)

# %%
criterion = nn.MSELoss()

# %%
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    num_epochs=EPOCHS,
    callbacks=[
        dl.R2SquaredCallback(input_key="logits", target_key="target"),
        dl.CheckpointCallback(logdir="checkpoints", loader_key="valid", metric_key="r2squared", topk=3, save_best=True, save_last=True, minimize=False),
        dl.CheckpointCallback(logdir="checkpoints_runner", save_last=True, save_best=False, mode="runner"),
        dl.SchedulerCallback()
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    cpu=False,
    fp16=False
)

# %%
runner.evaluate_loader(
    loader=loaders["valid"],
    callbacks=[
        dl.R2SquaredCallback(input_key="logits", target_key="target")
    ]
)


