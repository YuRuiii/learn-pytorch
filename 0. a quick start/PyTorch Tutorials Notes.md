#### PyTorch Tutorials

- task: build a classifier for fashion items
- load data: 2 primitives, data loader (an iterable over dataset) and dataset. Load dataset to data loader. Batch size for the number of elements returned each time.
- creating models: define child class of `nn.Module`, define layers in `__init__`, specify data passing in `forward`
- Optimizing the Model Parameters

  - to train a model, we need a loss fn and an optimizer. In a single training loop: predict -> compute error -> backpropagation to adjust parameters
  - also, do test each epoch to ensure accuracy increase and loss decrease
  - optimizer:
    - package: [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim)
    - basic: give it an iterable containing the parameters (`Variable`s) to optimize. then specify options like learning rate or weight decay.
    - to customize options for different parameters, pass `dict`s with different options instead
    - `optimizer.step()`: update the parameters after gradient computation like `backward()`. And for need of model re-computation, use `optimizer.step(closure)`. Here the closure should clear the gradients, compute the loss, and return it. 
    - lr decay: `torch.optim.lr_scheduler` provide methods for dynamic lr reducing. Lr scheduling should be applied after optimizer's update.
- Save and load model: by its state dict
- `with torch.no_grad()`: no need to calculate gradient. 1. when you only need to process forward propagation process, it's like detach all nodes in a model (graph) then run an epoch 2. when you don't want backward propagation to influence the nodes before a, detach it. (like in moba, we don't want gd to influence hero classifier): `torch.nn.functional.mse(a.detach(),b)`