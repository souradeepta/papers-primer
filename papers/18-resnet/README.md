# Deep Residual Learning for Image Recognition (ResNet)

## TL;DR

ResNet changes a stack of layers from “learn the whole desired mapping” to
“learn a correction to the input.” A shortcut carries the input around a small
convolutional branch and the two paths are added: \(y=F(x)+x\). That simple
reparameterization made very deep image networks substantially easier to
optimize. It does not mean deeper is automatically better, or that every
shortcut is an identity when shapes change.

## Why It Matters

By 2015, convolutional networks had shown that depth could improve image
recognition, but simply stacking more layers exposed a degradation problem: a
deeper plain network could have *higher training error* than a shallower one.
That is distinct from overfitting. In principle, added layers could implement
identity and preserve the shallower solution; in practice, the optimizer had
trouble finding it. ResNet gave every block an easy identity reference point.

The original paper evaluated networks up to 152 layers on ImageNet, describing
that model as eight times deeper than VGG while having lower complexity. Its
residual blocks became standard visual backbones and the core idea spread into
transformers, diffusion models, graph networks, and numerical-modeling views
of neural nets. Today `torchvision.models.resnet50` is a useful baseline, but
modern implementations vary in normalization order, stem, stride placement,
and training recipe from the 2015 paper.

## Core Intuition

Suppose a photo-processing pipeline already passes a useful image feature
forward. A new stage usually needs a small correction—sharpen this edge, add a
texture cue, suppress a background—not a complete replacement. Asking the
stage to output only its correction is easier than asking it to reconstruct the
entire signal. If no correction helps, the learned branch can approach zero and
the bypass still transports the signal.

```mermaid
flowchart LR
 X[input features x] --> F[convolutional residual branch F(x)]
 X --> S[identity shortcut]
 F --> A[add]
 S --> A
 A --> Y[output y]
```

This is not a claim that a residual branch literally learns an image residual
in pixel space. It is a feature-space parameterization. The shortcut also gives
gradients a direct additive route through a block, though successful deep
training still depends on initialization, normalization, data, and schedules.

## The Mechanism

A basic residual block defines \(y=F(x,\{W_i\})+x\), followed by the paper's
activation placement. For the two-layer example, \(F\) can be convolution,
batch normalization, ReLU, then another convolutional transform. Addition is
elementwise, so both paths must agree in spatial size and channel count. An
identity shortcut adds no learned parameters and almost no compute beyond the
addition.

```mermaid
flowchart TD
 X[x: H×W×C] --> C1[3×3 convolution + BN + ReLU]
 C1 --> C2[3×3 convolution + BN]
 X --> Add[add elementwise]
 C2 --> Add
 Add --> R[ReLU]
```

![Illustrative residual shortcut](assets/residual_shortcut.gif)

When a block changes resolution or channel count, its shortcut cannot be a
literal identity. The paper describes options including padding plus a stride,
or a learned projection (often a 1×1 convolution) \(W_sx\), giving
\(y=F(x)+W_sx\). Projection shortcuts cost parameters and compute, but align
shapes. The original ImageNet architectures used two 3×3 layers in basic blocks
for 18/34-layer models and 1×1, 3×3, 1×1 bottleneck blocks for 50/101/152-layer
models, reducing expensive wide 3×3 computation.

The paper trained ImageNet models with batch normalization after each
convolution and before activation, SGD with momentum, and data augmentation.
Those details matter: a shortcut alone is not a reproduction recipe. Later
“pre-activation” ResNets move normalization and activation before weight
layers; that is an influential refinement, not the exact original block.

The GIF is an illustrative teaching diagram, not a result from the paper. It
shows why zeroing the residual branch preserves a same-shaped input. In a real
network, ReLU and normalization affect exact identity behavior, and a learned
block need not have a small numerical residual. The empirical claim is about
easier optimization of the architecture family, not a universal proof.

## Practical Engineering Notes

Start with `torchvision.models.resnet18` or `resnet50` and inspect its weight
metadata, input normalization, and license rather than copying a checkpoint
name into production. A classifier head must match class order exactly; store
the labels, resize/crop convention, color order, and normalization constants
alongside weights. Fine-tuning only the head is cheap, but freezing a backbone
whose source images differ strongly from the target may cap accuracy.

Profile activation memory as well as parameters. Deeper networks retain more
intermediate tensors for backpropagation; gradient checkpointing trades extra
compute for memory. BatchNorm depends on training-mode batch statistics and
running estimates, so small per-device batches, distributed synchronization,
and accidental `train()`/`eval()` mode changes can dominate a debugging session.
For small batches, GroupNorm or carefully frozen BatchNorm may be appropriate,
but that changes the recipe and needs validation.

At serving time, fuse supported convolution/normalization operations, batch
requests within latency limits, and benchmark the actual input sizes. A ResNet
that is accurate at 224×224 can fail operationally if the crop discards the
object users care about. Test corruptions, class imbalance, calibration, and
slice performance; residual connections do not mitigate dataset bias. Use
feature-map and latency telemetry to catch an unintended resolution change.

### Training and integration decisions

Choose the block variant as part of the interface, not as an invisible
implementation detail. A checkpoint trained with a basic block cannot simply
load into a bottleneck architecture because tensor shapes and stage widths
differ. Likewise, changing the stride from the first to the second convolution
in a bottleneck changes feature alignment and may invalidate published weight
porting assumptions. Framework model factories encode these decisions; pin the
factory version and verify a known image's logits after export.

The additive operation requires disciplined tensor layout. In NCHW systems the
two tensors must agree on batch, channel, height, and width; broadcasting a
singleton dimension can silently create a different operation in generic array
code. Assert shapes at custom stage boundaries, especially after a branch adds
an attention module or a feature pyramid. For quantized inference, ensure the
two add inputs have compatible quantization scales or use the backend's fused
residual-add support; otherwise conversions can introduce unexpected latency or
accuracy movement.

Treat transfer learning as an experiment in representation reuse. Start with a
linear probe to measure whether frozen features carry target information, then
compare partial and full fine-tuning under identical augmentations. A small
target set can overfit the head while making a validation curve look smooth;
preserve a final test split and inspect confusion matrices. If labels describe
objects at a much smaller scale than ImageNet crops, adjust the resolution and
re-benchmark rather than assuming the architecture is at fault.

Residual feature extractors are often connected to detection and segmentation
necks. Those consumers depend on named stage outputs, spatial stride, and
channel counts. Document this feature contract and test it when replacing a
backbone. It prevents a common production failure where a seemingly compatible
new ResNet changes a pyramid level's geometry and degrades downstream boxes
without raising a shape error.

Finally, distinguish a residual architecture from a reproducible model result.
The paper's success used a particular dataset, augmentation, optimization
schedule, and evaluation protocol. Recreating the block in a modern framework
is valuable, but claiming its benchmark number requires reproducing the whole
measurement path, including preprocessing and multi-crop or multi-scale
inference where applicable.

## Runnable Code Example

[`code/residual_block.py`](code/residual_block.py) represents the add operation
with lists and asserts the essential invariant: a zero residual branch preserves
its input. Run it with:

```bash
python3 papers/18-resnet/code/residual_block.py
```

It is not a convolutional trainer; the small example isolates the architectural
contract that a framework block must satisfy before it handles shapes and
normalization.

## Common Misconceptions & Pitfalls

**“A shortcut removes vanishing gradients.”** It provides an easier additive
path but does not eliminate all optimization or numerical failures.

**“Residual means the network predicts image differences.”** The residual is a
learned mapping between internal feature tensors.

**“Every shortcut is free.”** Projection shortcuts, normalization, and shape
changes add cost and must be included in profiling.

## Interview Q&A

**Q:** What problem did ResNet target?
**A:** Optimization degradation: deeper plain networks could train worse than
shallower counterparts, even before considering test overfitting.

**Q:** Why can residual learning help?
**A:** It makes identity-like behavior easy: the branch can learn a correction
near zero while the shortcut carries the input.

**Q:** When is a projection shortcut needed?
**A:** When spatial resolution or channel count changes and elementwise addition
would otherwise have incompatible shapes.

**Q:** What is a bottleneck block?
**A:** A 1×1 reduction, 3×3 processing, and 1×1 expansion design that enables
deep, wide stages more efficiently.

**Q:** Is ResNet only for vision?
**A:** No. The residual parameterization is widely used wherever deep transforms
benefit from an identity reference path.

## Further Reading

- [Original paper](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [Torchvision ResNet documentation](https://pytorch.org/vision/stable/models/resnet.html)
