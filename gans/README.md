# Generative Adversarial Networks (GANs)

## Overview

Generative Adversarial Networks (GANs) are a class of machine learning frameworks introduced by Ian Goodfellow et al. in 2014. GANs consist of two neural networks—a generator and a discriminator—that compete against each other in a game-theoretic scenario.

## Key Concepts

### 1. Generator Network
- Creates synthetic data samples from random noise
- Learns to map from latent space to data space
- Tries to fool the discriminator into thinking its outputs are real

### 2. Discriminator Network
- Classifies samples as real or fake
- Learns to distinguish between real data and generated data
- Provides feedback to improve the generator

### 3. Adversarial Training
- Minimax game between generator and discriminator
- Both networks improve simultaneously
- Training converges when generator produces realistic samples

### 4. Loss Functions
- **Discriminator Loss**: Maximize correct classification
- **Generator Loss**: Minimize discriminator's ability to detect fakes
- Various formulations: vanilla GAN, WGAN, LSGAN, etc.

## Architecture Components

### Generator
- Input: Random noise vector (latent code)
- Layers: Dense/Convolutional layers with upsampling
- Output: Generated data (e.g., images)
- Activation: Tanh or Sigmoid for final layer

### Discriminator
- Input: Real or generated data
- Layers: Convolutional/Dense layers with downsampling
- Output: Probability (real vs. fake)
- Activation: Sigmoid for binary classification

## Mathematical Foundations

The GAN objective is a minimax game:

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Where:
- G: Generator network
- D: Discriminator network
- x: Real data samples
- z: Random noise (latent vector)
- D(x): Discriminator's estimate that x is real
- G(z): Generated sample from noise z

## Training Challenges

1. **Mode Collapse**: Generator produces limited variety
2. **Training Instability**: Oscillating losses, no convergence
3. **Vanishing Gradients**: Poor gradient flow to generator
4. **Evaluation Metrics**: Difficult to measure quality objectively

## Solutions and Improvements

- **Wasserstein GAN (WGAN)**: Uses Earth Mover's Distance
- **Spectral Normalization**: Stabilizes discriminator training
- **Progressive Growing**: Gradually increases resolution
- **Self-Attention**: Captures long-range dependencies
- **Conditional GANs**: Incorporates label information

## Applications

- **Image Generation**: Creating realistic images from scratch
- **Image-to-Image Translation**: Style transfer, domain adaptation
- **Data Augmentation**: Generating synthetic training data
- **Super-Resolution**: Enhancing image quality
- **Video Generation**: Creating synthetic video sequences
- **Text-to-Image**: Generating images from descriptions
- **Drug Discovery**: Molecular structure generation

## Learning Objectives

By the end of this module, you will:

1. Understand the adversarial training paradigm
2. Implement basic and advanced GAN architectures
3. Train GANs on various datasets (MNIST, CIFAR, custom data)
4. Apply techniques to stabilize GAN training
5. Use GANs for practical applications
6. Evaluate generated samples using quantitative metrics

## Prerequisites

- Deep Learning fundamentals
- Convolutional Neural Networks
- PyTorch or TensorFlow knowledge
- Understanding of backpropagation
- Basic probability and statistics

## Files in This Module

- `gans.ipynb`: Interactive Jupyter notebook with implementations
- `requirements.txt`: Python dependencies
- `gans.tex`: Mathematical foundations and detailed theory
- `data/`: Sample datasets and generated outputs

## GAN Variants

1. **DCGAN**: Deep Convolutional GAN
2. **WGAN**: Wasserstein GAN
3. **StyleGAN**: Style-based generator architecture
4. **CycleGAN**: Unpaired image-to-image translation
5. **Pix2Pix**: Paired image-to-image translation
6. **BigGAN**: Large-scale GAN training
7. **Progressive GAN**: Progressive growing of GANs

## References

1. Goodfellow, I., et al. (2014). "Generative Adversarial Nets"
2. Radford, A., et al. (2015). "Unsupervised Representation Learning with DCGANs"
3. Arjovsky, M., et al. (2017). "Wasserstein GAN"
4. Karras, T., et al. (2019). "A Style-Based Generator Architecture for GANs" (StyleGAN)
5. Zhu, J., et al. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Open `gans.ipynb` in Jupyter Lab or Notebook
3. Follow the progressive examples from basic to advanced GANs
4. Compile `gans.tex` for comprehensive mathematical explanations
