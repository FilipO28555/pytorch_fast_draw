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

```python
c = color(device='cuda')
c.red                      # predefined tensors: red, green, blue, white, ...
c.col(r, g, b)             # custom RGB tensor
c.colLuma(v)               # grayscale tensor
```

## License

MIT License — Copyright (c) 2025 Filip Optołowicz, University of Wrocław
