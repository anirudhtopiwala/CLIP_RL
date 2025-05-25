# CLIP Robot Control

## 1. Project Overview

This project demonstrates goal-conditioned robot control using CLIP embeddings for both visual observations and natural language instructions. The agent is trained in a simulated environment (Robosuite) to perform tasks such as "pick up the red cube" by mapping CLIP features to robot actions using reinforcement learning.

---

## 2. Installation

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd clip_robot_control
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **(Optional) Install MuJoCo and Robosuite:**
    - Follow [MuJoCo installation instructions](https://mujoco.org/) for your OS.
    - Install Robosuite:
      ```sh
      pip install robosuite
      ```

---

## 3. Training Details

- **Training Script:**  
  Run the following command to start training:
  ```sh
  python train_policy.py --num_episodes 1000 --save_path clip_policy.pt
  ```
  - Use `--model_path` to warm start from a checkpoint.
  - Training logs and metrics are saved for TensorBoard visualization.

- **Monitoring:**  
  Launch TensorBoard to monitor training:
  ```sh
  tensorboard --logdir runs
  ```

- **Environment:**  
  The agent is trained in Robosuite's "Lift" task using CLIP embeddings for both image and text goal.

- **Hardware:**  
  Training is optimized for GPU and uses mixed precision for speed.

---

For more details, see the code and comments in each script.