import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from Analyzer import analyze

class SatIntelTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("SatIntel Pro - Intelligence Dashboard")
        self.root.geometry("1550x950")
        self.root.configure(bg="#1c1c1c") 
        self.results_cache = {} 
        self.setup_ui()

    def setup_ui(self):
        # Header Section
        header = tk.Frame(self.root, bg="#00a8ff", pady=12)
        header.pack(fill="x")
        tk.Label(header, text="SATELLITE MULTI-CLASS ANALYSIS TERMINAL", 
                 font=("Arial", 20, "bold"), fg="white", bg="#00a8ff").pack()

        # Adjustable Paned Window
        self.paned = tk.PanedWindow(self.root, orient="horizontal", bg="#1c1c1c", sashwidth=6)
        self.paned.pack(expand=True, fill="both")

        # --- LEFT PANEL: Narrow Gallery Grid ---
        self.left_panel = tk.Frame(self.paned, bg="#252525", width=450)
        self.paned.add(self.left_panel)
        
        tk.Label(self.left_panel, text="SCANNED IMAGE GRID", fg="#7f8c8d", bg="#252525", font=("Arial", 9, "bold")).pack(pady=10)
        
        self.canvas = tk.Canvas(self.left_panel, bg="#252525", highlightthickness=0)
        self.scroll_y = ttk.Scrollbar(self.left_panel, orient="vertical", command=self.canvas.yview)
        self.grid_container = tk.Frame(self.canvas, bg="#252525")
        
        self.canvas.create_window((0, 0), window=self.grid_container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")
        self.grid_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # --- RIGHT PANEL: Wide Intelligence Workspace ---
        self.right_panel = tk.Frame(self.paned, bg="#1c1c1c", padx=30)
        self.paned.add(self.right_panel)

        self.res_title = tk.Label(self.right_panel, text="ANALYSIS: STANDBY", font=("Arial", 22, "bold"), fg="#00a8ff", bg="#1c1c1c")
        self.res_title.pack(pady=25)

        # Large Comparison View
        self.img_container = tk.Frame(self.right_panel, bg="#1c1c1c")
        self.img_container.pack(fill="x", pady=10)
        
        self.orig_view = tk.Label(self.img_container, bg="#252525", bd=2, relief="flat", text="ORIGINAL", fg="#7f8c8d")
        self.orig_view.pack(side="left", expand=True, padx=15)
        
        self.seg_view = tk.Label(self.img_container, bg="#252525", bd=2, relief="flat", text="SEGMENTED", fg="#7f8c8d")
        self.seg_view.pack(side="left", expand=True, padx=15)

        # Environmental Feature Table
        tk.Label(self.right_panel, text="DETAILED ENVIRONMENTAL ANALYSIS", 
                 fg="#7f8c8d", bg="#1c1c1c", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))
        
        style = ttk.Style()
        style.theme_use('clam') # Required to customize headers correctly
        style.configure("Treeview", rowheight=30, font=("Arial", 11), background="#252525", foreground="white", fieldbackground="#252525")
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"), background="#333333", foreground="white")

        self.tree = ttk.Treeview(self.right_panel, columns=("Feature", "Coverage"), show="headings", height=8)
        self.tree.heading("Feature", text="ENVIRONMENTAL FEATURE")
        self.tree.heading("Coverage", text="COVERAGE ANALYSIS (%)")
        self.tree.column("Feature", width=350)
        self.tree.column("Coverage", width=200)
        self.tree.pack(fill="x")

        # Command Button
        btn_frame = tk.Frame(self.right_panel, bg="#1c1c1c")
        btn_frame.pack(fill="x", pady=30)
        ttk.Button(btn_frame, text="IMPORT NEW DATASET", command=self.load_batch).pack(side="right")

    def load_batch(self):
        paths = filedialog.askopenfilenames()
        if not paths: return
        
        for widget in self.grid_container.winfo_children(): widget.destroy()

        col, row = 0, 0
        for path in paths:
            data = analyze(path)
            self.results_cache[path] = data
            
            tile = tk.Frame(self.grid_container, bg="#252525", padx=5, pady=5)
            tile.grid(row=row, column=col, padx=8, pady=8)
            
            img = ImageOps.fit(Image.open(path), (160, 160))
            tk_img = ImageTk.PhotoImage(img)
            
            btn = tk.Button(tile, image=tk_img, bg="#252525", activebackground="#00a8ff",
                            command=lambda p=path: self.display_data(p), borderwidth=0)
            btn.image = tk_img
            btn.pack()
            
            tk.Label(tile, text=data['label'], fg="#00a8ff", bg="#252525", font=("Arial", 8)).pack()
            
            col += 1
            if col > 1: # 2 Columns keeps the left panel narrow
                col = 0
                row += 1

    def display_data(self, path):
        data = self.results_cache[path]
        self.res_title.config(text=f"CLASSIFICATION: {data['label'].upper()}")
        
        display_size = (380, 380)
        o_img = ImageTk.PhotoImage(ImageOps.fit(Image.open(path), display_size))
        s_img = ImageTk.PhotoImage(ImageOps.fit(Image.open(data['seg_path']), display_size))
        
        self.orig_view.config(image=o_img); self.orig_view.image = o_img
        self.seg_view.config(image=s_img); self.seg_view.image = s_img

        for i in self.tree.get_children(): self.tree.delete(i)
        for line in data['report']:
            if ":" in line:
                self.tree.insert("", "end", values=line.split(": "))

if __name__ == "__main__":
    root = tk.Tk()
    app = SatIntelTerminal(root)
    root.mainloop()