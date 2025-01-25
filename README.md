<!-- # Restricted Boltzman Machine :

This is a personal implementation of a RBM-model, trained to do a generative-image task.

such models approxiamte the underlying distribution probability of a dataset.

We use binary images here, but the work is applicable to RGB-values images as well :

### Mathematical-Formulation

Let $V$ the random variable that underlies our data distribution that lives in the space $\set{0,1}^p$ \*we can see the image as a big vector where p = hxw$.
such distribution is by nature much difficult, and leverage multimodality behaviours, supposing some hard constraint about the distribution (i.e being some gaussian) is in vain.
In the generative AI formulations, authors propose to go through some intermediate hidden random variable $H$ that lives in a relatively other space $\set{0,1}^q$ when coupled to it, and explains $V$ since it's coupled to it.

For example : take the height distribution inside a population, the estimated distribution is a mono-modal Gaussian, which parameters aren't easy to estimate, but whith conditioning over the random variable $X(\omega) = \set{male, female} $, the distribution becomes an easy Gaussian, we estimate the parameters of each gaussian seperatly.

![Alt text](./imgs/height_dist.png)

Once $V$ is found, we can easily substitue the $p_V$ distrbution by marginaliziong over $p_H$. but such task isn't that easy for complexe phenomenon.

The RBM formulation relies on the following energy-based model :

$$p_{\theta}(h|v) = e^{-E_{\theta} (v,h)} / Z(\theta)$$

where $E\_\theta(v,h) = a^T v + b^T h +  v^T W h$ and $\theta = \set{a \in R^p, b \in R^q, W \in R^{p \times q}}$ the set of parameters to find.

This formulation benefits from such mathematical properties, as $h$ being conditionly independent to $v$, the inverse is also true, whoch helps for sampling later.

![Alt text](./imgs/rbm_states.png)

techniques such Gibbs-Sampling are used to sample from distribution to make it computationally feasible.

### Finale Note :

this relies purely on mathematicall results, and is implemented using only numpy package, isn't it interesting :D -->

# Restricted Boltzmann Machine (RBM)

This repository contains a personal implementation of a **Restricted Boltzmann Machine (RBM)** model, trained to perform generative tasks on binary images. While binary images are used here, the method can be extended to handle RGB images as well.

RBMs approximate the underlying probability distribution of a dataset, making them useful for various generative tasks in machine learning.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
- [RBM Energy-Based Model](#rbm-energy-based-model)
- [Sampling Techniques](#sampling-techniques)
- [Final Note](#finale-note)

---

## Overview

Restricted Boltzmann Machines are probabilistic graphical models that aim to learn the underlying data distribution. In this project, we implement and train an RBM to generate images from learned distributions. The core idea is to capture complex multimodal behaviors in the data without imposing strict assumptions like Gaussianity.

---

## Mathematical Formulation

Let $V$ be the random variable representing the underlying data distribution, which resides in the space $\{0,1\}^p$. In the context of images, $p = h \times w$, where we can view the image as a vector of pixels.

Approximating this distribution directly is challenging due to its multimodal nature. To simplify, we introduce an intermediate hidden random variable $H$ that lives in another space $\{0,1\}^q$. When coupled with $V$, it explains the variations in the data.

### Example:

Consider the height distribution within a population. The distribution may seem unimodal and Gaussian at first glance, but estimating its parameters may be difficult. However, by conditioning on the variable $X(\omega) = \{\text{male}, \text{female}\}$, the distribution becomes bimodal and much easier to model separately.

![Height Distribution](./imgs/height_dist.png)

Similarly, RBMs introduce hidden variables to make modeling more tractable. Once $V$ is discovered, we can marginalize over the hidden variable $H$ to approximate $p(V)$. However, for complex phenomena, this marginalization is computationally challenging.

---

## RBM Energy-Based Model

The RBM formulation relies on the following energy-based model:

$$ p*{\theta}(h|v) = \frac{e^{-E*{\theta}(v, h)}}{Z(\theta)} $$

Where the energy function $E_\theta(v, h)$ is defined as:

$$ E\_{\theta}(v,h) = a^T v + b^T h + v^T W h $$

Here:

- $a \in \mathbb{R}^p$ is the bias for the visible units.
- $b \in \mathbb{R}^q$ is the bias for the hidden units.
- $W \in \mathbb{R}^{p \times q}$ is the weight matrix connecting visible and hidden units.
- $\theta = \{a, b, W\}$ is the set of parameters to learn.

This energy-based formulation provides the property of conditional independence between $V$ and $H$, simplifying the learning and sampling processes.

---

## Sampling Techniques

Techniques such as **Gibbs Sampling** are used to sample from the RBM's distribution, making it computationally feasible to train and generate new data.

![RBM States](./imgs/rbm_states.png)

Gibbs sampling helps efficiently sample from the joint distribution, iteratively updating visible and hidden units based on their conditional probabilities.

---

## Finale Note

This project is purely based on mathematical principles and implemented using only the **NumPy** library. Isn't that fascinating? 😄

Feel free to explore the code and experiment with different datasets or configurations!

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
