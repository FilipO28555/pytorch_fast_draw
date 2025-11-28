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

class Canvas:
    def __init__(self, width, height, device='cuda'):
        """Initialize a canvas with specified dimensions."""
        self.width = width
        self.height = height
        self.device = device
        self.img = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)
        
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
            self.img[rows, cols] = color
            
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


# Usage example
if __name__ == "__main__":
    canvas = Canvas(800, 800)
    col = color().col
    import time
    lines = torch.randint(0, 800, (10_000, 4), device='cuda')
    points = torch.randint(0, 800, (1_000_000, 2), device='cuda')
    colors = col(255, 0, 0)
    colorsPoints = col(0, 255, 0)
    
    # Warmup - run each function once to compile CUDA kernels
    canvas.draw(points[:1000], colorsPoints)
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
    
    
    


