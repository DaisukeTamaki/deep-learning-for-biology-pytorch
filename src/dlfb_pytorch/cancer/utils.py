def split_decay_params(model):
  """Split model parameters into decay and no-decay groups.

  Excludes biases and normalization layer parameters from weight decay.
  """
  decay = []
  no_decay = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if _is_no_decay(name):
      no_decay.append(param)
    else:
      decay.append(param)
  return decay, no_decay


def build_param_groups(named_params, lr, weight_decay):
  """Build optimizer parameter groups with weight decay exclusion."""
  decay, no_decay = [], []
  for name, param in named_params:
    if not param.requires_grad:
      continue
    if _is_no_decay(name):
      no_decay.append(param)
    else:
      decay.append(param)
  groups = []
  if decay:
    groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
  if no_decay:
    groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
  return groups


def _is_no_decay(name: str) -> bool:
  name_lower = name.lower()
  return any(
    k in name_lower for k in ("bias", "bn", "batchnorm", "norm", "layernorm")
  )
