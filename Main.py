import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
from Analyzer import analyze

class SatIntelTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("SatIntel Enterprise - Grid Interface")
        self.root.state('zoomed')
        self.root.configure(bg="#080808")
        self.cache = {}
        self.setup_ui()

    def setup_ui(self):
        # --- SIDEBAR WITH SCROLLBAR & 2-COLUMN GRID ---
        sidebar_frame = tk.Frame(self.root, bg="#121212", width=300)
        sidebar_frame.pack(side="left", fill="y")
        sidebar_frame.pack_propagate(False)

        tk.Label(sidebar_frame, text="SATELLITE INVENTORY", bg="#121212", fg="#00d4ff", font=("Impact", 12)).pack(pady=15)

        # Scrollable Area Setup
        self.canvas_side = tk.Canvas(sidebar_frame, bg="#121212", highlightthickness=0, width=280)
        self.scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=self.canvas_side.yview)
        self.grid_container = tk.Frame(self.canvas_side, bg="#121212")

        # Window creation inside canvas
        self.canvas_window = self.canvas_side.create_window((0, 0), window=self.grid_container, anchor="nw")
        
        # Binding for dynamic scroll region
        self.grid_container.bind("<Configure>", self._on_frame_configure)
        self.canvas_side.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas_side.pack(side="left", fill="both", expand=True)

        # --- MAIN WORKSPACE ---
        self.work_area = tk.Frame(self.root, bg="#080808")
        self.work_area.pack(side="right", expand=True, fill="both")

        self.title_lbl = tk.Label(self.work_area, text="TERMINAL STANDBY", font=("Courier", 18, "bold"), bg="#080808", fg="#00ff88")
        self.title_lbl.pack(pady=10)

        # Image Viewports
        img_host = tk.Frame(self.work_area, bg="#080808")
        img_host.pack(pady=5)
        self.box_L = tk.Frame(img_host, width=450, height=450, bg="#111", bd=1, relief="solid")
        self.box_R = tk.Frame(img_host, width=450, height=450, bg="#111", bd=1, relief="solid")
        for b in [self.box_L, self.box_R]:
            b.pack_propagate(False)
            b.pack(side="left", padx=10)

        self.view_L = tk.Label(self.box_L, bg="#111")
        self.view_L.pack(expand=True, fill="both")
        self.view_R = tk.Label(self.box_R, bg="#111")
        self.view_R.pack(expand=True, fill="both")

        # Dual Tables
        report_frame = tk.Frame(self.work_area, bg="#080808")
        report_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.tree_model = ttk.Treeview(report_frame, columns=("M"), show="headings", height=5)
        self.tree_model.heading("M", text="VGG16 NEURAL LOG")
        self.tree_model.pack(side="left", fill="both", expand=True, padx=5)

        self.tree_feat = ttk.Treeview(report_frame, columns=("F"), show="headings", height=5)
        self.tree_feat.heading("F", text="SPECTRAL SEGMENTATION (%)")
        self.tree_feat.pack(side="left", fill="both", expand=True, padx=5)

        # Fixed Button
        btn_bar = tk.Frame(self.work_area, bg="#080808")
        btn_bar.pack(side="bottom", fill="x", pady=20)
        ttk.Button(btn_bar, text="IMPORT BATCH", command=self.import_imgs).pack(side="right", padx=50)

    def _on_frame_configure(self, event):
        self.canvas_side.configure(scrollregion=self.canvas_side.bbox("all"))

    def import_imgs(self):
        paths = filedialog.askopenfilenames()
        if not paths: return
        for w in self.grid_container.winfo_children(): w.destroy()
        
        for i, p in enumerate(paths):
            res = analyze(p)
            self.cache[p] = res
            tile = tk.Frame(self.grid_container, bg="#121212")
            tile.grid(row=i//2, column=i%2, padx=10, pady=10) # 2-Column Logic
            
            img = ImageOps.fit(Image.open(p), (110, 110))
            tk_img = ImageTk.PhotoImage(img)
            btn = tk.Button(tile, image=tk_img, command=lambda x=p: self.show(x), bg="#121212", bd=0)
            btn.image = tk_img; btn.pack()
            
            lbl_txt = os.path.basename(p)[:12]
            tk.Label(tile, text=lbl_txt, bg="#121212", fg="#555", font=("Arial", 7)).pack()

    def show(self, p):
        d = self.cache.get(p)
        if not d: return
        self.title_lbl.config(text=f"ANALYZING: {d['label'].upper()}")
        
        img1 = ImageTk.PhotoImage(ImageOps.fit(Image.open(p), (450, 450)))
        self.view_L.config(image=img1); self.view_L.image = img1
        
        img2 = ImageTk.PhotoImage(ImageOps.fit(Image.open(d['seg_path']), (450, 450)))
        self.view_R.config(image=img2); self.view_R.image = img2
            
        for tree, data_key in [(self.tree_model, 'model_data'), (self.tree_feat, 'feature_data')]:
            for i in tree.get_children(): tree.delete(i)
            for s in d[data_key]: tree.insert("", "end", values=(s,))

if __name__ == "__main__":
    app = SatIntelTerminal(tk.Tk()); tk.mainloop()