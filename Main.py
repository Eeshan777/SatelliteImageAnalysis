import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
from Analyzer import analyze

class SatIntelTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("SatIntel Pro - Intelligence Dashboard")
        self.root.state('zoomed') 
        self.root.configure(bg="#1c1c1c") 
        self.results_cache = {} 
        self.setup_ui()

    def setup_ui(self):
        # 1. Header
        header = tk.Frame(self.root, bg="#00a8ff", height=50)
        header.pack(fill="x")
        tk.Label(header, text="SATELLITE ANALYSIS TERMINAL", 
                 font=("Arial", 16, "bold"), fg="white", bg="#00a8ff").pack(pady=10)

        # 2. Main Layout
        self.paned = tk.PanedWindow(self.root, orient="horizontal", bg="#1c1c1c", sashwidth=4)
        self.paned.pack(expand=True, fill="both")

        # Sidebar (2 Columns)
        self.left_panel = tk.Frame(self.paned, bg="#252525", width=260) 
        self.paned.add(self.left_panel)
        
        tk.Label(self.left_panel, text="UPLOADED IMAGES", font=("Arial", 8, "bold"), 
                 fg="#7f8c8d", bg="#252525", pady=10).pack()
        
        self.canvas = tk.Canvas(self.left_panel, bg="#252525", highlightthickness=0)
        self.grid_container = tk.Frame(self.canvas, bg="#252525")
        self.canvas.create_window((0, 0), window=self.grid_container, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Main Workspace
        self.right_panel = tk.Frame(self.paned, bg="#1c1c1c")
        self.paned.add(self.right_panel)

        self.res_title = tk.Label(self.right_panel, text="STATUS: STANDBY", 
                                  font=("Arial", 20, "bold"), fg="#00a8ff", bg="#1c1c1c")
        self.res_title.pack(pady=15)

        # 3. Triple Image View (Equal Spacing)
        self.img_box = tk.Frame(self.right_panel, bg="#1c1c1c")
        self.img_box.pack(fill="both", expand=True, padx=20)
        
        for i in range(3): self.img_box.columnconfigure(i, weight=1)
        self.img_box.rowconfigure(0, weight=1)

        self.orig_view = tk.Label(self.img_box, bg="#252525")
        self.orig_view.grid(row=0, column=0, sticky="nsew", padx=8)
        
        self.recon_view = tk.Label(self.img_box, bg="#252525")
        self.recon_view.grid(row=0, column=1, sticky="nsew", padx=8)
        
        self.seg_view = tk.Label(self.img_box, bg="#252525")
        self.seg_view.grid(row=0, column=2, sticky="nsew", padx=8)

        # 4. Data Table & Button
        self.bottom_frame = tk.Frame(self.right_panel, bg="#1c1c1c")
        self.bottom_frame.pack(fill="x", side="bottom", padx=30, pady=20)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", background="#252525", foreground="white", fieldbackground="#252525")
        
        self.tree = ttk.Treeview(self.bottom_frame, columns=("F", "C"), show="headings", height=5)
        self.tree.heading("F", text="ENVIRONMENTAL FEATURE"); self.tree.heading("C", text="COVERAGE %")
        self.tree.column("F", width=350); self.tree.column("C", width=120)
        self.tree.pack(fill="x")

        ttk.Button(self.bottom_frame, text="IMPORT DATASET", command=self.load_batch).pack(side="right")

    def load_batch(self):
        paths = filedialog.askopenfilenames()
        if not paths: return
        for widget in self.grid_container.winfo_children(): widget.destroy()
        
        col, row = 0, 0
        for index, path in enumerate(paths):
            data = analyze(path)
            self.results_cache[path] = data
            
            tile = tk.Frame(self.grid_container, bg="#252525", pady=5)
            tile.grid(row=row, column=col, padx=10, pady=8)
            
            # FIT to square for consistent sidebar
            img = ImageOps.fit(Image.open(path), (100, 100), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            
            btn = tk.Button(tile, image=tk_img, bg="#252525", bd=0, command=lambda p=path: self.display_data(p))
            btn.image = tk_img
            btn.pack()
            
            tk.Label(tile, text=f"IMAGE {index}", font=("Arial", 7, "bold"), fg="#00a8ff", bg="#252525").pack()
            
            col += 1
            if col > 1: col = 0; row += 1

    def display_data(self, path):
        data = self.results_cache.get(path)
        if not data: return
        self.res_title.config(text=f"DETECTED: {data['label'].upper()}", fg="#2ecc71")
        
        self.root.update_idletasks()
        
        # Calculate target based on the gray boxes
        target_w = self.orig_view.winfo_width()
        target_h = self.orig_view.winfo_height()

        image_streams = [
            (path, self.orig_view),
            (data.get('recon_path'), self.recon_view),
            (data.get('seg_path'), self.seg_view)
        ]

        for img_path, widget in image_streams:
            if img_path and os.path.exists(img_path):
                # Use FIT with centering to ensure the images fill the boxes correctly
                img = ImageOps.fit(Image.open(img_path), (target_w, target_h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                widget.config(image=tk_img)
                widget.image = tk_img

        for i in self.tree.get_children(): self.tree.delete(i)
        for line in data.get('report', []):
            if ":" in line: self.tree.insert("", "end", values=line.split(": "))

if __name__ == "__main__":
    root = tk.Tk(); app = SatIntelTerminal(root); root.mainloop()