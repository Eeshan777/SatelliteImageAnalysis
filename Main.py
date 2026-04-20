import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
from Analyzer import analyze

class SatIntelTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("SatIntel Enterprise - 10-Class Spectral Interface")
        self.root.state('zoomed')
        self.root.configure(bg="#080808")
        self.cache = {}
        self.img_refs = {}
        self.setup_ui()

    def setup_ui(self):
        # --- SIDEBAR (Scrollable) ---
        side = tk.Frame(self.root, bg="#121212", width=280)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self.canvas = tk.Canvas(side, bg="#121212", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(side, orient="vertical", command=self.canvas.yview)
        self.grid_area = tk.Frame(self.canvas, bg="#121212")
        self.canvas.create_window((0,0), window=self.grid_area, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_area.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # --- WORKSPACE ---
        work = tk.Frame(self.root, bg="#080808")
        work.pack(side="right", fill="both", expand=True)

        self.title_lbl = tk.Label(work, text="TERMINAL READY", font=("Courier", 18), bg="#080808", fg="#00ff88")
        self.title_lbl.pack(pady=10)

        # Button Bar
        btn_bar = tk.Frame(work, bg="#121212", height=60)
        btn_bar.pack(side="bottom", fill="x")
        btn_bar.pack_propagate(False)
        ttk.Button(btn_bar, text="IMPORT BATCH", command=self.import_imgs).pack(side="right", padx=20, pady=15)

        # Images
        img_p = tk.Frame(work, bg="#080808")
        img_p.pack(pady=5)
        self.view_L = tk.Label(img_p, width=380, height=380, bg="#111", bd=1, relief="solid")
        self.view_R = tk.Label(img_p, width=380, height=380, bg="#111", bd=1, relief="solid")
        self.view_L.pack(side="left", padx=10)
        self.view_R.pack(side="left", padx=10)

        # --- TABLES ---
        table_p = tk.Frame(work, bg="#080808")
        table_p.pack(fill="both", expand=True, padx=20, pady=10)

        # Neural Table
        self.tree_m = ttk.Treeview(table_p, columns=("L", "D", "C"), show="headings", height=8)
        for c, h in zip(("L", "D", "C"), ("Location", "Detection", "Confidence")):
            self.tree_m.heading(c, text=h); self.tree_m.column(c, width=100, anchor="center")
        self.tree_m.pack(side="left", fill="both", expand=True, padx=5)

        # Spectral Table with Legend Column
        self.tree_f = ttk.Treeview(table_p, columns=("F", "P", "CLR"), show="headings", height=8)
        self.tree_f.heading("F", text="Feature")
        self.tree_f.heading("P", text="Coverage")
        self.tree_f.heading("CLR", text="Legend")
        self.tree_f.column("CLR", width=50, anchor="center")
        self.tree_f.pack(side="left", fill="both", expand=True, padx=5)

    def import_imgs(self):
        paths = filedialog.askopenfilenames()
        if not paths: return
        for w in self.grid_area.winfo_children(): w.destroy()
        for i, p in enumerate(paths):
            res = analyze(p)
            self.cache[p] = res
            img = ImageTk.PhotoImage(ImageOps.fit(Image.open(p), (110, 110)))
            b = tk.Button(self.grid_area, image=img, command=lambda x=p: self.show(x), bg="#121212", bd=0)
            b.image = img; b.grid(row=i//2, column=i%2, padx=5, pady=5)

    def show(self, p):
        d = self.cache.get(p)
        if not d: return
        self.title_lbl.config(text=f"ANALYSIS: {d['label'].upper()}")
        
        # Update Images
        self.img_refs['L'] = ImageTk.PhotoImage(ImageOps.fit(Image.open(p), (380, 380)))
        self.view_L.config(image=self.img_refs['L'])
        if os.path.exists(d['seg_path']):
            self.img_refs['R'] = ImageTk.PhotoImage(ImageOps.fit(Image.open(d['seg_path']), (380, 380)))
            self.view_R.config(image=self.img_refs['R'])

        # Update Neural Table
        for i in self.tree_m.get_children(): self.tree_m.delete(i)
        for row in d['model_data']: self.tree_m.insert("", "end", values=row)

        # Update Spectral Table with Colors
        for i in self.tree_f.get_children(): self.tree_f.delete(i)
        for name, perc, color in d['feature_data']:
            tag = f"tag_{name.replace(' ', '')}"
            self.tree_f.insert("", "end", values=(name, perc, "■"), tags=(tag,))
            self.tree_f.tag_configure(tag, foreground=color)

if __name__ == "__main__":
    app = SatIntelTerminal(tk.Tk()); tk.mainloop()