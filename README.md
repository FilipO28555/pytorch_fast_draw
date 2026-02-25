# pytorch_fast_draw

GPU-accelerated drawing utilities using PyTorch.

## Requirements

- PyTorch (CUDA recommended) — [pytorch.org](https://pytorch.org/get-started/locally/)
- `pip install opencv-python numpy`

## Canvas API

| Method | Description |
|--------|-------------|
| `Canvas(w, h, device)` | Create a canvas |
| `clear(col=None)` | Fill with color or black |
| `draw(pos, col)` | Draw points. `pos`: `(N,2)`, `col`: `(3,)` or `(N,3)` |
| `add(pos, col)` | Like `draw` but accumulates colors on overlap |
| `drawLine(pos, col)` | Draw lines. `pos`: `(N,4)` as `(x0,y0,x1,y1)` |
| `drawLineGradient(pos, col_start, col_end)` | Lines with per-line color gradient |
| `drawHistogram(data, bin_width, ...)` | Histogram from raw 1-D tensor values |
| `drawBarChart(counts, ...)` | Bar chart where each element is a bar height directly |
| `display(wait=1)` | Show via OpenCV. Returns `True` on ESC |
| `save_png(path)` | Save to file |
| `setTitle(title)` | Set window title |

### `drawHistogram` options
```python
canvas.drawHistogram(
    data,            # 1-D tensor
    bin_width=10,
    data_range=(0, 100),   # optional, inferred from data if omitted
    col=..., bg_col=..., padding=40, bar_gap=1,
    title='My histogram', x_label='Value', y_label='Count'
)
```

### `drawBarChart` options
```python
canvas.drawBarChart(
    counts,          # 1-D tensor — values are bar heights directly
    col=..., bg_col=..., padding=50, bar_gap=1,
    title='My chart', x_label='Bin', y_label='Probability'
)
```

Both plotting functions draw a background grid with Y-scale numbers and X-axis tick labels automatically.

## color helper

All color values are `torch.uint8` tensors on the chosen device with shape `(3,)` = `[R, G, B]`.

```python
c = color(device='cuda')
```

**Predefined colors** — all `torch.uint8` tensors, shape `(3,)`:

| Attribute | RGB |
|-----------|-----|
| `c.white` | 255, 255, 255 |
| `c.black` | 0, 0, 0 |
| `c.red` | 255, 0, 0 |
| `c.green` | 0, 255, 0 |
| `c.blue` | 0, 0, 255 |
| `c.yellow` | 255, 255, 0 |
| `c.magenta` | 255, 0, 255 |
| `c.cyan` | 0, 255, 255 |
| `c.gray` | 128, 128, 128 |
| `c.dark_gray` | 64, 64, 64 |
| `c.light_gray` | 192, 192, 192 |

**Methods:**

```python
c.col(r, g, b)    # → torch.uint8 tensor [r, g, b], values 0–255
c.colLuma(v)      # → torch.uint8 tensor [v, v, v]
                  #   v can be a scalar (int/float) or a tensor
                  #   (returns shape (3,) or (N,3) respectively)
```

Colors are accepted by all drawing methods as either:
- **`(3,)`** — single color applied to all primitives
- **`(N, 3)`** — per-primitive color, one row per point/line

## License

MIT License — Copyright (c) 2025 Filip Optołowicz, University of Wrocław
