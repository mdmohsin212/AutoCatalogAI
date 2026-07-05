from contextlib import nullcontext
import torch
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from autocatalog.training.losses import compute_multitask_loss
from autocatalog.utils.logger import get_logger
logger = get_logger(__name__)


def _autocast(device, enabled):
    if not enabled or not str(device).startswith("cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def _scaler(device, enabled):
    enabled = enabled and str(device).startswith("cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_stage(model, stage_name, stage_config):
    for parameter in model.parameters():
        parameter.requires_grad = False

    new_modules = [
        model.master_to_sub,
        model.sub_to_article,
        model.article_to_season,
        model.article_to_usage,
        model.color_branch,
    ]

    for module in new_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    active_tasks = stage_config["active_tasks"]
    for task in active_tasks:
        for parameter in model.heads[task].parameters():
            parameter.requires_grad = True

    if stage_name == "stage2":
        vision_layers = model.clip.vision_model.encoder.layers
        count = stage_config["unfreeze_last_n_vision_layers"]

        for layer in vision_layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        for parameter in model.clip.visual_projection.parameters():
            parameter.requires_grad = True

        for parameter in model.clip.vision_model.post_layernorm.parameters():
            parameter.requires_grad = True

    return active_tasks, new_modules


def create_optimizer_and_scheduler(model, stage_name, stage_config, weight_decay, steps_per_epoch, device, use_amp):
    active_tasks, new_modules = configure_stage(model, stage_name,stage_config)
    new_parameter_ids = {
        id(parameter)
        for module in new_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }

    head_parameters = []
    new_parameters = []
    backbone_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in new_parameter_ids:
            new_parameters.append(parameter)
        elif name.startswith("heads"):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    groups = []
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": stage_config["head_lr"],
            }
        )

    if new_parameters:
        groups.append(
            {
                "params": new_parameters,
                "lr": stage_config["new_layer_lr"],
            }
        )

    if backbone_parameters and "backbone_lr" in stage_config:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": stage_config["backbone_lr"],
            }
        )

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=weight_decay,
    )

    total_steps = stage_config["epochs"] * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.10)),
        num_training_steps=total_steps,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info("%s ready | active_tasks=%s | trainable_parameters=%s", stage_name, active_tasks, f"{trainable_parameters:,}")
    return (
        active_tasks,
        optimizer,
        scheduler,
        _scaler(device, use_amp),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterions,
    active_tasks,
    device,
    use_amp,
    max_grad_norm,
):
    model.train()
    total_loss = 0.0
    sample_count = 0

    correct = {
        task: 0
        for task in model.heads.keys()
    }

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        color_features = batch["color_features"].to(device)

        labels = {
            task: tensor.to(device)
            for task, tensor in batch["labels"].items()
        }

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(
                pixel_values,
                color_features,
            )

            loss = compute_multitask_loss(
                outputs,
                labels,
                criterions,
                active_tasks,
            )

        if use_amp and str(device).startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            optimizer.step()
            
        scheduler.step()
        batch_size = pixel_values.size(0)
        sample_count += batch_size
        total_loss += loss.item() * batch_size

        for task in correct:
            predictions = outputs[task].argmax(dim=1)

            correct[task] += int(
                (predictions == labels[task]).sum().item()
            )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            color_acc=f"{correct['baseColour'] / sample_count:.4f}",
        )

    average_loss = total_loss / max(sample_count, 1)
    accuracies = {
        task: count / max(sample_count, 1)
        for task, count in correct.items()
    }

    return average_loss, accuracies