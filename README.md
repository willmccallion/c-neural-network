# Neural Network Visualizer in C

A Convolutional Neural Network (CNN) implemented from scratch in C99 without external machine learning libraries. This project demonstrates the low-level implementation of deep learning architectures, including convolution operations, pooling layers, and the Adam optimizer. It features a custom merged dataset of 70 classes (digits, letters, and drawings) and visualizes internal network states in real-time using Raylib.

## Project Overview

The objective is to implement a functional deep learning engine at the memory level, managing raw pointers, tensor arithmetic, and thread synchronization manually.

**Key Technical Features:**
*   **Custom CNN Architecture:** Implementation of convolution (3x3 kernels), max pooling, and fully connected layers.
*   **Parallel Processing:** Uses OpenMP for matrix operation acceleration and Pthreads to separate the training loop from the UI rendering loop.
*   **Adam Optimizer:** Implementation of Adaptive Moment Estimation for weight updates.
*   **Automated Data Pipeline:** CMake and Python integration to automatically download, normalize, and merge EMNIST and QuickDraw datasets.
*   **Real-Time Visualization:** Renders activation heatmaps, weight histograms, and loss/accuracy graphs during active training.

## Architecture

The network uses a 3-stage convolutional architecture designed for the 70-class merged dataset:

1.  **Block 1:** Conv2D (16 filters, 3x3) → ReLU → MaxPool (2x2).
2.  **Block 2:** Conv2D (32 filters, 3x3) → ReLU → MaxPool (2x2).
3.  **Block 3:** Conv2D (64 filters, 3x3) → ReLU → MaxPool (2x2).
4.  **Dense Layer:** Flatten → 256 Hidden Nodes (ReLU).
5.  **Output Layer:** 70 Nodes (Softmax).

## Build Instructions

This project uses CMake for build configuration. It automatically handles C dependencies (Raylib) and Python dependencies (for data generation).

### Prerequisites
*   C Compiler (GCC/Clang/MSVC) with OpenMP support.
*   CMake 3.24 or higher.
*   Python 3.x (for data generation script).

### Compilation

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Configure the project:
    ```bash
    cmake ..
    ```
    *Note: The first run will create a Python virtual environment and download the datasets. This may take a few minutes.*

3.  Build the executable:
    ```bash
    make
    ```
    *(Or `cmake --build .` on Windows)*

## Data Setup

Data acquisition is automated. The build system triggers `scripts/data.py`, which:
1.  Downloads the EMNIST dataset (Balanced split).
2.  Downloads specific categories from the Google QuickDraw dataset.
3.  Normalizes orientation and merges them into a single binary format.
4.  Outputs `extended-train-images-idx3-ubyte` and `extended-train-labels-idx1-ubyte` into the `build/data` directory.

## Usage

Run the executable from the build directory or the project root.

```bash
./draw_predictor
```

### Interface Modes

The application is divided into two tabs:

1.  **Dashboard:**
    *   **Canvas:** Draw digits, letters, or specific shapes (Apple, Book, Car, etc.).
    *   **Robot Vision:** Shows the downscaled (28x28) and centered input seen by the network.
    *   **Network View:** Visualizes the active feature maps in real-time.
    *   **Predictions:** Displays the top 5 confidence scores.

2.  **Training Analytics:**
    *   **Live Feed:** Shows the images currently being processed by the training thread.
    *   **Graphs:** Real-time plotting of Loss and Validation Accuracy.
    *   **Heatmaps:** Visualizes the activation patterns of all three convolutional layers.
    *   **Histograms:** Displays the distribution of weights across layers to monitor for vanishing/exploding gradients.

### Controls

*   **Left Click:** Draw on canvas / Switch tabs / Toggle training pause.
*   **Right Click / 'C':** Clear the canvas.

## Implementation Details

### Threading Model
The application uses a producer-consumer pattern with a mutex lock.
*   **Main Thread:** Handles Raylib rendering, input processing, and visualization. It reads from shared state buffers.
*   **Training Thread:** Runs the training loop (forward/backward pass) or inference loop (idle mode). It updates the shared state buffers and synchronizes via `pthread_mutex`.

### Forward Propagation
The convolution operation is implemented manually using 3x3 sliding windows. The output of each filter is passed through a Rectified Linear Unit (ReLU) activation function before being downsampled by a 2x2 Max Pooling operation.

### Backpropagation
Gradients are calculated using the chain rule.
*   **Dense Layers:** Standard matrix multiplication gradients.
*   **Pooling Layers:** Gradients are passed only to the index of the maximum value from the forward pass.
*   **Convolution Layers:** Gradients are calculated by convolving the error map with the rotated weights (full convolution).

### Optimization
The Adam optimizer maintains moment estimates (mean and uncentered variance) for every weight and bias in the network to adapt the learning rate for each parameter individually.
