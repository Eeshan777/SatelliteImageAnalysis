import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
from Analyzer import analyze

class SatIntelTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("SatIntel Enterprise - Precision Analysis Terminal")
        self.root.state('zoomed')
        self.root.configure(bg="#080808")
        self.cache = {}
        self.img_refs = {}
        
        # --- DARK THEME STYLE CONFIG ---
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#121212", foreground="#e0e0e0", 
                             fieldbackground="#121212", rowheight=25, borderwidth=0)
        self.style.configure("Treeview.Heading", background="#1a1a1a", foreground="#00ff88", borderwidth=1)
        self.style.map("Treeview", background=[('selected', '#004433')])

        self.setup_ui()

    def setup_ui(self):
        # Sidebar
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

        # Workspace
        work = tk.Frame(self.root, bg="#080808")
        work.pack(side="right", fill="both", expand=True)

        self.title_lbl = tk.Label(work, text="MATRIX READY", font=("Courier", 18), bg="#080808", fg="#00ff88")
        self.title_lbl.pack(pady=10)

        # Button Bar
        btn_bar = tk.Frame(work, bg="#121212", height=60)
        btn_bar.pack(side="bottom", fill="x")
        btn_bar.pack_propagate(False)
        ttk.Button(btn_bar, text="IMPORT BATCH", command=self.import_imgs).pack(side="right", padx=20, pady=15)

        # Image Panel
        img_p = tk.Frame(work, bg="#080808")
        img_p.pack(pady=5)
        self.view_L = tk.Label(img_p, width=400, height=400, bg="#000", bd=1, relief="solid")
        self.view_R = tk.Label(img_p, width=400, height=400, bg="#000", bd=1, relief="solid")
        self.view_L.pack(side="left", padx=10)
        self.view_R.pack(side="left", padx=10)

        # --- TABLES (ALIGNMENT FIX) ---
        table_p = tk.Frame(work, bg="#080808")
        table_p.pack(fill="both", expand=True, padx=20, pady=10)
        table_p.columnconfigure(0, weight=1)
        table_p.columnconfigure(1, weight=1)

        # Neural Matrix
        self.tree_m = ttk.Treeview(table_p, columns=("L", "D", "C"), show="headings", height=12)
        for c, h in zip(("L", "D", "C"), ("Loc", "Detection", "Conf")):
            self.tree_m.heading(c, text=h); self.tree_m.column(c, width=100, anchor="center")
        self.tree_m.grid(row=0, column=0, sticky="nsew", padx=5)

        # Spectral Legend
        self.tree_f = ttk.Treeview(table_p, columns=("F", "P", "K"), show="headings", height=12)
        for c, h in zip(("F", "P", "K"), ("Feature", "Coverage", "Key")):
            self.tree_f.heading(c, text=h)
        self.tree_f.column("K", width=50, anchor="center")
        self.tree_f.grid(row=0, column=1, sticky="nsew", padx=5)

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
        self.title_lbl.config(text=f"5x5 MATRIX ANALYSIS: {d['label'].upper()}")
        
        self.img_refs['L'] = ImageTk.PhotoImage(ImageOps.fit(Image.open(p), (400, 400)))
        self.view_L.config(image=self.img_refs['L'])
        if os.path.exists(d['seg_path']):
            self.img_refs['R'] = ImageTk.PhotoImage(ImageOps.fit(Image.open(d['seg_path']), (400, 400)))
            self.view_R.config(image=self.img_refs['R'])

        for i in self.tree_m.get_children(): self.tree_m.delete(i)
        for row in d['model_data']: self.tree_m.insert("", "end", values=row)

        for i in self.tree_f.get_children(): self.tree_f.delete(i)
        for name, perc, color in d['feature_data']:
            tag = f"tag_{name.replace(' ', '')}"
            self.tree_f.insert("", "end", values=(name, perc, "■"), tags=(tag,))
            self.tree_f.tag_configure(tag, foreground=color)

if __name__ == "__main__":
    app = SatIntelTerminal(tk.Tk()); tk.mainloop()