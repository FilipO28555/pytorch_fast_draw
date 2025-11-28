# PytorchFastDraw

GPU-accelerated drawing utilities using PyTorch for fast, parallel rendering operations.

## Overview

PytorchFastDraw provides a `Canvas` class that leverages PyTorch tensors and CUDA acceleration to perform high-performance drawing operations. This is particularly useful for applications requiring real-time rendering of many primitives simultaneously.

## Features

- **GPU Acceleration**: All drawing operations run on CUDA-enabled GPUs
- **Batch Operations**: Draw thousands of points and lines in parallel
- **Gradient Lines**: Support for color gradient line rendering
- **OpenCV Integration**: Easy display and export via OpenCV

## Requirements

- Python 3.x
- PyTorch (with CUDA support recommended)
- OpenCV (`cv2`)

## Installation
Install torch with CUDA support following the instructions at [PyTorch.org](https://pytorch.org/get-started/locally/), then install the required Python packages:
```bash
pip install opencv-python
```

## Usage

```python
from torch_draw import Canvas

# Create a canvas (800x600 pixels on GPU)
canvas = Canvas(800, 600, device='cuda')

# Clear with a color
canvas.clear(torch.tensor([0, 0, 0], device='cuda'))

# Draw points
positions = torch.tensor([[100, 100], [200, 200]], device='cuda')
color = torch.tensor([255, 0, 0], dtype=torch.uint8, device='cuda')
canvas.draw(positions, color)

# Draw two lines (x0, y0, x1, y1)
lines = torch.tensor([[0, 0, 100, 100], [50, 0, 50, 100]], device='cuda')
canvas.drawLine(lines, color)

# Draw gradient lines
canvas.drawLineGradient(
    lines,
    col_start=torch.tensor([255, 0, 0], dtype=torch.uint8, device='cuda'),
    col_end=torch.tensor([0, 0, 255], dtype=torch.uint8, device='cuda')
)

# Display the result
canvas.display()
```

## License

MIT License - Copyright (c) 2025 Filip Optołowicz, University of Wrocław
