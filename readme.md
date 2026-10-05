# Representation Learning via Composite Subspace Flow Matching

In this work, we introduce **Composite Subspace Flow Matching (CSFM)**, a generative framework that repurposes continuous normalizing flows for self-supervised representation learning. 

### Abstract
>Distilling semantic meaning from raw sensory data remains a central challenge in machine learning. 
>Current self-supervised paradigms face a fundamental trade-off: joint-embedding architectures produce relevant features by aggressively discarding intra-class variance, which irreversibly destroys spatial topology and precludes reconstruction. 
>Conversely, generative models preserve spatial details but struggle to isolate semantics, as their optimization is dominated by low-level pixel variations.
>To bridge this divide, we propose composite subspace flow matching (CSFM). 
>Rather than corrupting data with unstructured isotropic noise, CSFM formulates representation learning as the variance-minimizing inversion of a deterministic degradation curriculum across spectral, color, and illumination subspaces. 
>By bounding trajectory variance with hard temporal style truncation, a capacity-constrained encoder is provably driven to ignore transient style artifacts and encode persistent content. 
>Empirical evaluations across CIFAR-10, CIFAR-100, and STL-10 show that CSFM yields competitive discriminative transferability while preserving generative synthesis across convolutional and vision transformer architectures.
>By avoiding the collapse pathologies of non-generative methods, the CSFM forms a smooth, high-rank latent space that consistently outperforms standard optimal transport flows on linear probing and few-shot classification while enabling coherent generative synthesis and trajectory interpolation.
>These results confirm that steering the physical dynamics of generative flows offers a scalable, principled paradigm for learning representations that are simultaneously linearly separable and generatively invertible.


## Codebase Overview

For researchers and reviewers looking for our specific contributions, please refer to the following files:
* `target_flow_lib.py`: Contains the physics engine for our Decoupled Optimal Transport and CSFM degradation paths.
* `losses.py`: Implements our core objectives, including the CSFM flow matching loss with Asymmetric Gradient Routing, alongside baselines (CDAE, LeJEA, AE).
* `run_lib.py`: The main orchestration file managing the training loops, EMA updates, and distributed optimization.


## Usage

### Full Pipeline (Train, Eval, Sample)

To run a complete CSFM model experiment (which automatically cycles through training, evaluation, and sampling based on your config intervals), execute:

Bash

```
python main.py --mode=all --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example
```

### Isolated Evaluation or Sampling

If you already have a trained checkpoint and want to evaluate features or generate new samples, you can run those modes in isolation:

Bash

```
# For feature evaluation (saves to features.h5)
python main.py --mode=eval --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example --checkpoint=path/to/checkpoint.pth

# For generating samples
python main.py --mode=sample --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example --checkpoint=path/to/checkpoint.pth
```

*All outputs, including model checkpoints, HDF5 embeddings, and generated image grids, will be saved automatically to the directory specified in `--workdir`.*