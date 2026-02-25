"""
torch_draw.py - GPU-accelerated drawing utilities using PyTorch

MIT License

Copyright (c) 2025 Filip Optołowicz
University of Wrocław
Contact: filipoptolowicz@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import cv2
import numpy as np

class Canvas:
    def __init__(self, width, height, device='cuda'):
        """Initialize a canvas with specified dimensions."""
        self.width = width
        self.height = height
        self.device = device
        self.img = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)
    def setTitle(self, title):
        """Set the window title for display."""
        cv2.setWindowTitle("Canvas", title)
        return self
    
    def clear(self, col=None):
        """Clear the canvas with the specified color."""
        if col is None:
            self.img = torch.zeros((self.height, self.width, 3), dtype=torch.uint8, device=self.device)
        else:
            self.img[:, :] = col
        return self
    
    def draw(self, pos, col):
        """Draw points at the specified positions with the specified colors.
        
        Args:
            pos: Tensor of shape (N, 2) containing the (x, y) coordinates
            col: Tensor of shape (3,) or (N, 3) containing RGB color(s)
        """
        # Apply mask for points within bounds
        mask_x = pos[:, 0].ge(0) & pos[:, 0].lt(self.width)
        mask_y = pos[:, 1].ge(0) & pos[:, 1].lt(self.height)
        valid_mask = mask_x & mask_y
        
        if not valid_mask.any():
            return self
            
        valid_pos = pos[valid_mask]
        
        # Handle color tensor shape
        if len(col.shape) == 1:
            # Single color for all points
            color = col
        else:
            # Different color for each point
            color = col[valid_mask] if valid_mask.any() else col
        
        # Use efficient batch-wise drawing
        rows = valid_pos[:, 1].long()
        cols = valid_pos[:, 0].long()
        
        if len(col.shape) == 1:
            self.img[rows, cols] = color
        else:
            # Multiple colors: assign colors per point
            self.img[rows, cols] = color

        return self
        
    def add(self, pos, col):
        """Draw points at the specified positions with the specified colors. Add colors to existing pixel values.
        Accumulates color for repeated points using flattening and index_add_.
        Args:
            pos: Tensor of shape (N, 2) containing the (x, y) coordinates
            col: Tensor of shape (3,) or (N, 3) containing RGB color(s)
        """
        # Apply mask for points within bounds
        mask_x = pos[:, 0].ge(0) & pos[:, 0].lt(self.width)
        mask_y = pos[:, 1].ge(0) & pos[:, 1].lt(self.height)
        valid_mask = mask_x & mask_y
        if not valid_mask.any():
            return self
        valid_pos = pos[valid_mask]
        # Handle color tensor shape
        if len(col.shape) == 1:
            color = col.unsqueeze(0).expand(valid_pos.shape[0], 3)
        else:
            color = col[valid_mask] if valid_mask.any() else col
        # Flatten image
        img_flat = self.img.view(-1, 3)
        rows = valid_pos[:, 1].long()
        cols = valid_pos[:, 0].long()
        flat_idx = rows * self.width + cols
        img_flat.index_add_(0, flat_idx, color)
        # Unflatten
        self.img = img_flat.view(self.height, self.width, 3)
        return self
    
    def drawLine(self, pos, col=None):
        """Draw lines using fully parallel computation with index_put_.
        
        Args:
            pos: Tensor of shape (N, 4) containing (x0, y0, x1, y1) for each line
            col: Tensor of shape (3,) containing RGB color
        """
        if col is None:
            col = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        if pos.shape[0] == 0:
            return self
        
        x0, y0, x1, y1 = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
        
        # Calculate max steps needed
        dx = (x1 - x0).abs()
        dy = (y1 - y0).abs()
        steps = torch.maximum(dx, dy)
        max_steps = int(steps.max().item()) + 1
        n_lines = pos.shape[0]
        
        # Create interpolation parameter t: shape (max_steps,)
        t = torch.arange(max_steps, device=self.device, dtype=torch.float32)
        
        # Broadcast and compute all coordinates
        # steps_clamped: (n_lines, 1), t: (1, max_steps) -> result: (n_lines, max_steps)
        steps_f = steps.float().clamp(min=1).unsqueeze(1)
        frac = t.unsqueeze(0) / steps_f  # (n_lines, max_steps)
        
        all_x = x0.unsqueeze(1) + ((x1 - x0).float().unsqueeze(1) * frac).long()
        all_y = y0.unsqueeze(1) + ((y1 - y0).float().unsqueeze(1) * frac).long()
        
        # Validity mask: within line length and canvas bounds
        valid = (t.unsqueeze(0) <= steps.unsqueeze(1)) & \
                (all_x >= 0) & (all_x < self.width) & \
                (all_y >= 0) & (all_y < self.height)
        
        # Extract valid coordinates
        y_coords = all_y[valid]
        x_coords = all_x[valid]
        
        # Direct indexing assignment
        self.img[y_coords, x_coords] = col
        
        return self
    
    def drawLineGradient(self, pos, col_start, col_end):
        """Draw lines with color gradient using fully parallel computation.
        
        Args:
            pos: Tensor of shape (N, 4) containing (x0, y0, x1, y1) for each line
            col_start: Tensor of shape (3,) or (N, 3) containing start RGB color(s)
            col_end: Tensor of shape (3,) or (N, 3) containing end RGB color(s)
        """
        if pos.shape[0] == 0:
            return self
        
        n_lines = pos.shape[0]
        x0, y0, x1, y1 = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
        
        # Calculate max steps needed
        dx = (x1 - x0).abs()
        dy = (y1 - y0).abs()
        steps = torch.maximum(dx, dy)
        max_steps = int(steps.max().item()) + 1
        
        # Create interpolation parameter t: shape (max_steps,)
        t = torch.arange(max_steps, device=self.device, dtype=torch.float32)
        
        # Broadcast and compute all coordinates
        steps_f = steps.float().clamp(min=1).unsqueeze(1)
        frac = t.unsqueeze(0) / steps_f  # (n_lines, max_steps)
        
        all_x = x0.unsqueeze(1) + ((x1 - x0).float().unsqueeze(1) * frac).long()
        all_y = y0.unsqueeze(1) + ((y1 - y0).float().unsqueeze(1) * frac).long()
        
        # Handle color tensors - expand to (n_lines, 3) if needed
        if len(col_start.shape) == 1:
            col_start = col_start.unsqueeze(0).expand(n_lines, 3)
        if len(col_end.shape) == 1:
            col_end = col_end.unsqueeze(0).expand(n_lines, 3)
        
        # Interpolate colors: (n_lines, max_steps, 3)
        # frac: (n_lines, max_steps) -> (n_lines, max_steps, 1)
        # col_start: (n_lines, 3) -> (n_lines, 1, 3)
        frac_color = frac.unsqueeze(2)  # (n_lines, max_steps, 1)
        col_start_f = col_start.float().unsqueeze(1)  # (n_lines, 1, 3)
        col_end_f = col_end.float().unsqueeze(1)  # (n_lines, 1, 3)
        
        all_colors = (col_start_f + (col_end_f - col_start_f) * frac_color).to(torch.uint8)
        
        # Validity mask: within line length and canvas bounds
        valid = (t.unsqueeze(0) <= steps.unsqueeze(1)) & \
                (all_x >= 0) & (all_x < self.width) & \
                (all_y >= 0) & (all_y < self.height)
        
        # Extract valid coordinates and colors
        y_coords = all_y[valid]
        x_coords = all_x[valid]
        colors = all_colors[valid]
        
        # Direct indexing assignment with per-pixel colors
        self.img[y_coords, x_coords] = colors
        
        return self
    
    def getImg(self):
        """Return the canvas as a numpy array."""
        return self.img.cpu().numpy()
        
    def display(self, wait=1):
        """Display the canvas using OpenCV.
        
        Args:
            wait: Time to wait in ms (0=wait forever)
            
        Returns:
            True if ESC key was pressed, False otherwise
        """
        img = self.getImg()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Canvas', img)
        key = cv2.waitKey(wait)
        return key == 27  # Return True if ESC pressed

    def save_png(self, path):
        """Save the canvas as a PNG file."""
        img = self.getImg()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    # ------------------------------------------------------------------
    # Internal helper: draw grid lines + axis tick labels onto the canvas
    # ------------------------------------------------------------------
    def _drawPlotGrid(self, plot_x0, plot_y0, plot_x1, plot_y1,
                      x_labels,        # list of (pixel_x, label_str) for x axis
                      y_values,        # list of (data_value, label_str) for y axis
                      max_val,         # data value that maps to plot_y0 (top)
                      n_grid_lines=5,
                      grid_col=(50, 50, 50),
                      axis_col=(100, 100, 100),
                      text_col=(180, 180, 180),
                      font_scale=0.35,
                      font=cv2.FONT_HERSHEY_SIMPLEX,
                      title=None,
                      x_label=None,
                      y_label=None):
        """Draw grid lines, axis tick labels, title and axis labels using OpenCV."""
        img_np = self.img.cpu().numpy().copy()   # RGB uint8
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        plot_h = plot_y1 - plot_y0
        title_scale   = font_scale * 1.6
        axlabel_scale = font_scale * 1.2

        # --- Title (centered above plot area) --------------------------------
        if title:
            (tw, th), _ = cv2.getTextSize(title, font, title_scale, 1)
            tx = max(0, (plot_x0 + plot_x1) // 2 - tw // 2)
            ty = max(th + 2, plot_y0 - 6)
            cv2.putText(img_bgr, title, (tx, ty), font, title_scale,
                        (220, 220, 220), 1, cv2.LINE_AA)

        # --- Pre-measure widest Y label so we can always fit it --------------
        y_labels = []
        for i in range(n_grid_lines + 1):
            frac = i / n_grid_lines
            val = frac * max_val
            y_labels.append(f'{val:.3g}')
        max_label_w = max(cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in y_labels)

        # --- Horizontal grid lines + Y tick labels ---------------------------
        for i in range(n_grid_lines + 1):
            frac = i / n_grid_lines
            y = int(plot_y1 - frac * plot_h)
            label = y_labels[i]
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.line(img_bgr, (plot_x0, y), (plot_x1, y), grid_col, 1)
            tx = max(2, plot_x0 - tw - 4)
            ty = max(th, min(y + th // 2, self.height - 2))
            cv2.putText(img_bgr, label, (tx, ty), font, font_scale, text_col, 1, cv2.LINE_AA)

        # --- Vertical axis line & bottom axis line ---------------------------
        cv2.line(img_bgr, (plot_x0, plot_y0), (plot_x0, plot_y1), axis_col, 1)
        cv2.line(img_bgr, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1)

        # --- X-axis tick labels ----------------------------------------------
        prev_right = -1
        x_tick_bottom = plot_y1   # track lowest y used by tick labels
        for (px, label) in x_labels:
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            tx = max(0, min(px - tw // 2, self.width - tw - 1))
            ty = min(plot_y1 + th + 4, self.height - 2)
            x_tick_bottom = max(x_tick_bottom, ty)
            if tx > prev_right:
                cv2.putText(img_bgr, label, (tx, ty), font, font_scale, text_col, 1, cv2.LINE_AA)
                prev_right = tx + tw + 2

        # --- X-axis label (centered below tick labels) -----------------------
        if x_label:
            (lw, lh), _ = cv2.getTextSize(x_label, font, axlabel_scale, 1)
            tx = max(0, (plot_x0 + plot_x1) // 2 - lw // 2)
            ty = min(x_tick_bottom + lh + 6, self.height - 2)
            cv2.putText(img_bgr, x_label, (tx, ty), font, axlabel_scale,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # --- Y-axis label (rotated 90°, centered on the left) ----------------
        if y_label:
            (lw, lh), _ = cv2.getTextSize(y_label, font, axlabel_scale, 1)
            # render onto a small temporary image, then rotate and blit
            tmp = np.zeros((lh + 4, lw + 4, 3), dtype=np.uint8)
            cv2.putText(tmp, y_label, (2, lh + 1), font, axlabel_scale,
                        (200, 200, 200), 1, cv2.LINE_AA)
            tmp_rot = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # position: vertically centered on the plot, touching the left edge
            rh, rw = tmp_rot.shape[:2]
            ry = max(0, (plot_y0 + plot_y1) // 2 - rh // 2)
            rx = max(0, 0)   # left edge of canvas
            ry2 = min(ry + rh, self.height)
            rx2 = min(rx + rw, self.width)
            roi = img_bgr[ry:ry2, rx:rx2]
            src = tmp_rot[:ry2-ry, :rx2-rx]
            mask = src.any(axis=2)
            roi[mask] = src[mask]
            img_bgr[ry:ry2, rx:rx2] = roi

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.img = torch.from_numpy(img_rgb).to(self.device)
        return self

    def drawHistogram(self, data, bin_width, data_range=None, col=None, bg_col=None,
                      padding=40, bar_gap=1, title=None, x_label=None, y_label=None):
        """Draw a histogram of the given data tensor on the canvas.

        Args:
            data:       1-D tensor of values to histogram.
            bin_width:  Scalar – width of each histogram bin (in data units).
            data_range: Optional tuple (min, max) specifying the data range to cover.
            col:        Bar fill color, shape (3,). Defaults to white.
            bg_col:     Background color, shape (3,). If None the canvas is not cleared.
            padding:    Pixel padding around the plot area (int).
            bar_gap:    Gap in pixels between adjacent bars (int, ≥0).
            title:      Optional title string drawn above the plot.
            x_label:    Optional x-axis label string.
            y_label:    Optional y-axis label string (drawn rotated on the left).
        """
        if col is None:
            col = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        if bg_col is not None:
            self.clear(bg_col)

        data = data.float().to(self.device)

        if data_range is not None:
            data_min, data_max = float(data_range[0]), float(data_range[1])
        else:
            data_min = data.min().item()
            data_max = data.max().item()

        if data_max == data_min:
            return self  # nothing to draw

        # --- Bin edges & counts ------------------------------------------------
        n_bins = max(1, int(round((data_max - data_min) / bin_width)))
        range_max = data_min + n_bins * bin_width
        # torch.histc needs float min/max
        counts = torch.histc(data, bins=n_bins,
                             min=float(data_min),
                             max=float(range_max))
        max_count = counts.max().item()
        if max_count == 0:
            return self

        # --- Plot area ---------------------------------------------------------
        plot_x0 = padding
        plot_y0 = padding
        plot_x1 = self.width  - padding
        plot_y1 = self.height - padding
        plot_w  = plot_x1 - plot_x0
        plot_h  = plot_y1 - plot_y0

        bar_w = max(1, plot_w // n_bins)

        # --- Grid & labels ----------------------------------------------------
        # X labels: one tick per bin, but skip to avoid crowding
        x_labels = []
        max_x_labels = max(1, plot_w // 40)
        step = max(1, n_bins // max_x_labels)
        for i in range(0, n_bins + 1, step):
            px = plot_x0 + i * bar_w
            val = data_min + i * bin_width
            x_labels.append((px, f'{val:.4g}'))
        self._drawPlotGrid(plot_x0, plot_y0, plot_x1, plot_y1,
                           x_labels=x_labels,
                           y_values=None,
                           max_val=max_count,
                           title=title, x_label=x_label, y_label=y_label)

        # --- Build all bar rectangles as line segments -------------------------
        # Each bar is filled by drawing (bar_h) horizontal lines.
        # We collect them as (x0,y0,x1,y1) segments and call drawLine once.
        all_lines = []
        for i in range(n_bins):
            bar_h = int(round(counts[i].item() / max_count * plot_h))
            if bar_h == 0:
                continue
            bx0 = plot_x0 + i * bar_w + bar_gap
            bx1 = plot_x0 + (i + 1) * bar_w - bar_gap - 1
            if bx1 < bx0:
                bx1 = bx0
            by1 = plot_y1          # bottom of bar (canvas y increases downward)
            by0 = plot_y1 - bar_h  # top of bar

            # One horizontal line per pixel row of the bar
            ys = torch.arange(by0, by1, device=self.device)
            x0s = torch.full_like(ys, bx0)
            x1s = torch.full_like(ys, bx1)
            segs = torch.stack([x0s, ys, x1s, ys], dim=1)  # (bar_h, 4)
            all_lines.append(segs)

        if not all_lines:
            return self

        all_lines = torch.cat(all_lines, dim=0)
        self.drawLine(all_lines, col)
        return self

    def drawBarChart(self, counts, col=None, bg_col=None, padding=50, bar_gap=1,
                     title=None, x_label=None, y_label=None):
        """Draw a bar chart where each element of counts is directly the bar height.

        Args:
            counts:  1-D tensor (or list) of non-negative values, one per bar.
            col:     Bar fill color, shape (3,). Defaults to white.
            bg_col:  Background color, shape (3,). If None the canvas is not cleared.
            padding: Pixel padding around the plot area (int).
            bar_gap: Gap in pixels between adjacent bars (int, ≥0).
            title:   Optional title string drawn above the plot.
            x_label: Optional x-axis label string.
            y_label: Optional y-axis label string (drawn rotated on the left).
        """
        if col is None:
            col = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        if bg_col is not None:
            self.clear(bg_col)

        counts = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        n_bars = counts.shape[0]
        max_count = counts.max().item()
        if max_count == 0 or n_bars == 0:
            return self

        # --- Plot area ---------------------------------------------------------
        plot_x0 = padding
        plot_y0 = padding
        plot_y1 = self.height - padding
        plot_w  = self.width  - 2 * padding
        plot_h  = self.height - 2 * padding

        bar_w = max(1, plot_w // n_bars)

        # --- Grid & labels ----------------------------------------------------
        max_x_labels = max(1, plot_w // 40)
        step = max(1, n_bars // max_x_labels)
        x_labels = []
        for i in range(0, n_bars + 1, step):
            px = plot_x0 + i * bar_w
            x_labels.append((px, str(i)))
        self._drawPlotGrid(plot_x0, plot_y0, self.width - padding, plot_y1,
                           x_labels=x_labels,
                           y_values=None,
                           max_val=max_count,
                           title=title, x_label=x_label, y_label=y_label)

        # --- Build all bar segments in one vectorised pass ---------------------
        # bar pixel heights: (n_bars,)
        bar_heights = (counts / max_count * plot_h).round().long().clamp(min=0)

        # For each bar, generate one horizontal segment per pixel row.
        # We do this fully on GPU without a Python loop over bars.

        # x extents per bar: (n_bars,)
        bx0s = plot_x0 + torch.arange(n_bars, device=self.device) * bar_w + bar_gap
        bx1s = (plot_x0 + (torch.arange(n_bars, device=self.device) + 1) * bar_w
                - bar_gap - 1).clamp(min=bx0s)

        # max bar height in pixels
        max_h = int(bar_heights.max().item())
        if max_h == 0:
            return self

        # pixel offsets from top of each bar: (n_bars, max_h)
        offsets = torch.arange(max_h, device=self.device).unsqueeze(0)  # (1, max_h)
        valid = offsets < bar_heights.unsqueeze(1)                       # (n_bars, max_h)

        # y coordinates: plot_y1 - bar_height + offset  => fills upward
        ys = (plot_y1 - bar_heights.unsqueeze(1) + offsets)  # (n_bars, max_h)

        # expand x extents
        x0s = bx0s.unsqueeze(1).expand_as(ys)
        x1s = bx1s.unsqueeze(1).expand_as(ys)

        # flatten and filter
        valid_flat = valid.reshape(-1)
        segs = torch.stack([
            x0s.reshape(-1)[valid_flat],
            ys.reshape(-1)[valid_flat],
            x1s.reshape(-1)[valid_flat],
            ys.reshape(-1)[valid_flat],
        ], dim=1)

        self.drawLine(segs, col)
        return self

class color:
    def __init__(self, device='cuda'):
        self.device = device
        self.white = torch.tensor([255, 255, 255], dtype=torch.uint8, device=device)
        self.black = torch.tensor([0, 0, 0], dtype=torch.uint8, device=device)
        self.red = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)
        self.green = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)
        self.blue = torch.tensor([0, 0, 255], dtype=torch.uint8, device=device)
        self.yellow = torch.tensor([255, 255, 0], dtype=torch.uint8, device=device)
        self.magenta = torch.tensor([255, 0, 255], dtype=torch.uint8, device=device)
        self.cyan = torch.tensor([0, 255, 255], dtype=torch.uint8, device=device)
        self.gray = torch.tensor([128, 128, 128], dtype=torch.uint8, device=device)
        self.dark_gray = torch.tensor([64, 64, 64], dtype=torch.uint8, device=device)
        self.light_gray = torch.tensor([192, 192, 192], dtype=torch.uint8, device=device)
        
    
    def col(self, r,g,b):
        return torch.tensor([r,g,b], dtype=torch.uint8, device=self.device)
    
    def colLuma(self, luma):
        # if luma is a number
        if isinstance(luma, (int, float)):
            return torch.tensor([luma, luma, luma], dtype=torch.uint8, device=self.device)
        # if luma is a tensor
        luma = torch.tensor(luma, device=self.device)
        return torch.stack([luma, luma, luma], dim=-1).to(torch.uint8)


# Usage example
if __name__ == "__main__":
    # Test the add method with repeated points
    canvas = Canvas(800, 800)
    col = color().col
    
    # There are two drawing functions: draw() and add()
    # draw() - draws points, if points are repeated, last color is used
    # add() - adds points, if points are repeated, all colors are used. Colour values are clamped to 255.
    
    # Points: Three circles regions with overlapping points
    radious = 800/4
    center1 = torch.tensor([200, 400], device='cuda')
    center2 = torch.tensor([400, 400], device='cuda')
    center3 = torch.tensor([600, 400], device='cuda')
    y, x = torch.meshgrid(torch.arange(800, device='cuda'), torch.arange(800, device='cuda'), indexing='ij')
    dist1 = torch.sqrt((x - center1[0])**2 + (y - center1[1])**2)
    dist2 = torch.sqrt((x - center2[0])**2 + (y - center2[1])**2)
    dist3 = torch.sqrt((x - center3[0])**2 + (y - center3[1])**2)
    mask1 = dist1 <= radious
    mask2 = dist2 <= radious
    mask3 = dist3 <= radious
    points1 = torch.stack([x[mask1], y[mask1]], dim=-1)
    points2 = torch.stack([x[mask2], y[mask2]], dim=-1)
    points3 = torch.stack([x[mask3], y[mask3]], dim=-1)
    points = torch.cat([points1, points2, points3], dim=0)
    colors1 = col(255, 0, 0).unsqueeze(0).expand(points1.shape[0], 3)
    colors2 = col(0, 255, 0).unsqueeze(0).expand(points2.shape[0], 3)
    colors3 = col(0, 0, 255).unsqueeze(0).expand(points3.shape[0], 3)
    colors = torch.cat([colors1, colors2, colors3], dim=0)
    canvas.add(points, colors)
    canvas.display(0)
    
    # PERFORMANCE TESTING
    
    canvas = Canvas(800, 800)
    col = color().col
    import time
    lines = torch.randint(0, 800, (10_000, 4), device='cuda')
    points = torch.randint(0, 800, (1_000_000, 2), device='cuda')
    colors = col(255, 0, 0)
    colorsPoints = torch.randint(0, 255, (1_000_000, 3), device='cuda').to(torch.uint8)
    # colorsPoints = col(0, 255, 0)
    
    # Warmup - run each function once to compile CUDA kernels
    canvas.draw(points[:1000], colorsPoints[:1000])
    canvas.drawLine(lines[:10], colors)
    canvas.drawLineGradient(lines[:10], col(255,255,255), col(0,0,0))
    canvas.clear()
    torch.cuda.synchronize()
    
    # Time for drawing points
    torch.cuda.synchronize()
    start_points = time.time()
    canvas.draw(points, colorsPoints)
    torch.cuda.synchronize()
    time_points = time.time() - start_points
    print("Time for drawing points:", time_points)
    
    #time for drawing points with add
    canvas.clear()
    torch.cuda.synchronize()
    start_points_add = time.time()
    canvas.add(points, colorsPoints)
    torch.cuda.synchronize()
    time_points_add = time.time() - start_points_add
    print("Time for drawing points with add:", time_points_add)
    
    # Time for drawing lines
    torch.cuda.synchronize()
    start_lines = time.time()
    canvas.drawLine(lines, colors)
    torch.cuda.synchronize()
    time_lines = time.time() - start_lines
    print("Time for drawing lines:", time_lines)
    
    # Time for drawing gradient lines
    col_start = col(255, 255, 255)  # White
    col_end = col(0, 0, 0)    # Black
    torch.cuda.synchronize()
    start_grad = time.time()
    canvas.drawLineGradient(lines, col_start, col_end)
    torch.cuda.synchronize()
    time_grad = time.time() - start_grad
    print("Time for drawing gradient lines:", time_grad)
    
    canvas.display(0)
    cv2.destroyAllWindows()
    
    
    # HISTOGRAM TESTING
    canvas = Canvas(800, 600)
    data = torch.cat([
        torch.randn(50_000, device='cuda') * 50 + 300,   # peak ~300
        torch.randn(20_000, device='cuda') * 30 + 600,   # peak ~600
    ])
    canvas.drawHistogram(data, bin_width=10,
                         col=col(100, 200, 255),
                         bg_col=torch.tensor([20, 20, 20], dtype=torch.uint8, device='cuda'),
                         padding=40)
    canvas.display(0)
    cv2.destroyAllWindows()

    # Histogram with explicit range: natural numbers 0-39, bin width 1
    canvas = Canvas(800, 400)
    data_int = torch.randint(0, 40, (100_000,), device='cuda').float()
    canvas.drawHistogram(data_int, bin_width=1, data_range=(0, 40),
                         col=col(255, 180, 0),
                         bg_col=torch.tensor([20, 20, 20], dtype=torch.uint8, device='cuda'),
                         padding=40)
    canvas.display(0)
    cv2.destroyAllWindows()

    # ANIMATED TESTING
    # wait for key
    cv2.waitKey(0)
    canvas = Canvas(800, 800)
    N_lines = 10_000
    N_points = 1_000_000
    while canvas.display(1) == False:
        lines = torch.randint(100, 700, (N_lines, 4), device='cuda')
        points = torch.randint(0, 800, (N_points, 2), device='cuda')
        colors = col(255, 0, 0)
        colorsPoints = col(0, 255, 0)
        canvas.clear().draw(points, colorsPoints).drawLineGradient(lines, col(255,255,255), col(0,0,0))





