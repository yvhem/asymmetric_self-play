# Asymmetric Self-Play for Automatic Goal Discovery in Robotic Manipulation

This repository contains a **PyTorch-based re-implementation** of OpenAI's framework for asymmetric self-play in robotic manipulation, based on the paper [*Asymmetric Self-Play for Automatic Goal Discovery in Robotic Manipulation* (OpenAI et al., 2021)](https://arxiv.org/abs/2101.04882). This project was developed as part of the *Reinforcement Learning* course (A.Y. 2024/2025) at Sapienza University of Rome.

## Overview

The primary challenge in multi-task robotic manipulation is training a single goal-conditioned policy capable of solving a wide variety of tasks without manual, hand-designed reward shaping or human-defined curricula. 

This framework solves this problem by training two competitive reinforcement learning agents in a table-top environment:
* **Alice (Goal Setter):** Interacts with objects on a table. The final state of her trajectory defines the goal configuration. Alice is rewarded if Bob fails to solve the goal she proposes, encouraging her to find progressively harder challenges.
* **Bob (Goal Solver):** A goal-conditioned policy trained to match the final object configuration set by Alice. Bob is rewarded for successfully reaching the goal state.

Through this competitive interaction, an **automatic curriculum** naturally emerges. Alice discovers complex manipulation tasks (such as picking, flipping, and stacking), while Bob progressively learns to master them.

---

## Key Features

1. **Goal-Conditioned Reinforcement Learning:** Trained using Proximal Policy Optimization (PPO) with generalized advantage estimation (GAE).
2. **Alice Behavioral Cloning (ABC):** Since Bob is trained with sparse rewards, random exploration on complex tasks (like stacking) would almost always fail. When Bob fails, Alice’s successful trajectory is relabeled as a demonstration, and Bob learns from it via behavioral cloning with PPO-style loss clipping.
3. **Permutation Invariant Network Architecture:** The policy and value networks process an arbitrary number of objects using a permutation-invariant embedding layer (via max-pooling), allowing the model to generalize to varying object counts.
4. **Memory-Enhanced Policies:** LSTM layers are integrated into the actor-critic heads to let Bob internalize environmental physical dynamics over multi-step episodes.
5. **Parallel Training via Ray:** Scalable parallel rollout workers execute self-play episodes concurrently, feeding centralized PPO update steps.

---

## Architecture

```
                       [ Input Layer ]
                              │
  ┌───────────────────────────┼───────────────────────────┐
  ▼                           ▼                           ▼
[ Robot Joint Position ]  [ Gripper Position ]     [ Object & Goal States ]
  │                           │                           │
  ▼                           ▼                           ▼
Embedding (256)            Embedding (256)         Permutation Invariant
  │                           │                       Embedding (512)
  └───────────────────────────┼───────────────────────────┘
                              ▼
                         [ Concatenate ]
                              │
                              ▼
                         [ MLP (512) ]
                              │
                              ▼
                         [ LSTM (256) ]
                              │
                              ▼
                    [ Actor & Critic Heads ]
```

* **Goal-Conditioning:** Bob's network includes both object and goal states. Alice’s network shares the same architecture but masks out the goal states with zero-padding as she is not goal-conditioned.
* **Permutation Invariance:** The object state concatenations are mapped through an MLP and pooled across the object dimension using a max-pooling operation, ensuring the order of objects on the table does not affect the policy output.

---

## File Structure

```text
├── models.py          # Actor-Critic network with Permutation Invariant Embeddings and LSTM
├── agent.py           # PPO Agent with GAE, rollout buffers, and ABC loss clipping
├── config.py          # Hyperparameters (PPO, self-play, rewards, and hardware)
├── utils.py           # Observation processing, padding, and GAE calculation
├── main.py            # Ray parallel training loop and rollouts
├── evaluate.py        # Evaluation on manual holdout tasks
├── requirements.txt   # Dependencies
└── models/            # Directory where trained checkpoints are stored
```

---

## Installation

### Prerequisites
* Python 3.7+
* [MuJoCo](https://github.com/google-deepmind/mujoco) (requires `mujoco-py` setup)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yvhem/asymmetric_self-play.git
   cd asymmetric_self-play
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This automatically installs OpenAI's `robogym` as an editable package.*

---

## Usage

### Training

To start training Alice and Bob using the default configuration (12 rollout worker CPUs, parallelized with Ray):
```bash
python main.py
```

To run a lightweight 2-worker debug session with real-time MuJoCo visualization:
```bash
python main.py --debug --num-objects 2
```

### Evaluation on Holdout Tasks

Once trained, Bob can be tested on a suite of **holdout tasks** that he was never explicitly trained on. This evaluates Bob's **zero-shot generalization** capabilities.

Available tasks: `push`, `flip`, `pick-and-place`, `stack`.

Run the evaluation script by loading a saved model checkpoint (e.g., at step `1000`):
```bash
python evaluate.py --task stack --load-step 1000 --episodes 10
```

---

## Results & Zero-Shot Generalization

When trained purely through asymmetric self-play, Bob successfully generalizes to several manually designed holdout tasks, demonstrating high success rates without manual curriculum engineering:

* **Push:** Placing blocks in target 2D poses.
* **Flip:** Rotating blocks to align specified faces.
* **Pick-and-place:** Lifting blocks and placing them in target 3D air coordinates.
* **Stack:** Constructing towers of multiple blocks.

Moreover, the self-play game setup encourages Alice to discover novel, creative goals (such as lifting multiple blocks at once, tilting held blocks, or balancing a 4-block tower), prompting Bob to learn creative solutions to solve them.

---

## References

* OpenAI, Plappert, M., Sampedro, R., Xu, T., Akkaya, I., Kosaraju, V., Welinder, P., D'Sa, R., Petron, A., Pinto, H. P. d. O., Paino, A., Noh, H., Weng, L., Yuan, Q., Chu, C., & Zaremba, W. (2021). *Asymmetric Self-Play for Automatic Goal Discovery in Robotic Manipulation.* [arXiv:2101.04882](https://arxiv.org/abs/2101.04882).
