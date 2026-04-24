import os
import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image,ImageTk,ImageOps
from Analyzer import analyze

class SatIntelTerminal:
	def __init__(self,root):
		self.root=root
		self.root.title("SatIntel Enterprise - Precision Analysis Terminal")
		self.root.state('zoomed')
		self.root.configure(bg="#080808")
		self.cache,self.img_refs,self.active_path={}, {},None
		self.current_grid=6
		self.style=ttk.Style()
		self.style.theme_use("clam")
		self.style.configure("Treeview",background="#121212",foreground="#e0e0e0",fieldbackground="#121212",rowheight=28)
		self.style.configure("Treeview.Heading",background="#1a1a1a",foreground="#00ff88")
		self.style.configure("Vertical.TScrollbar",gripcount=0,background="#333",arrowcolor="#00ff88")
		self.setup_ui()
	def setup_ui(self):
		side_L=tk.Frame(self.root,bg="#121212",width=280)
		side_L.pack(side="left",fill="y")
		side_L.pack_propagate(False)
		tk.Button(side_L,text="UPLOAD IMAGES",bg="#00ff88",fg="#000",font=("Courier",10,"bold"),command=self.import_files,height=2).pack(fill="x",padx=10,pady=10)
		list_container=tk.Frame(side_L,bg="#121212")
		list_container.pack(fill="both",expand=True)
		self.canvas_L=tk.Canvas(list_container,bg="#121212",highlightthickness=0,width=250)
		self.scroll_L=ttk.Scrollbar(list_container,orient="vertical",command=self.canvas_L.yview)
		self.grid_area=tk.Frame(self.canvas_L,bg="#121212")
		self.canvas_window=self.canvas_L.create_window((0,0),window=self.grid_area,anchor="nw")
		self.grid_area.bind("<Configure>",self.on_frame_configure)
		self.canvas_L.configure(yscrollcommand=self.scroll_L.set)
		self.canvas_L.pack(side="left",fill="both",expand=True)
		self.scroll_L.pack(side="right",fill="y")
		self.canvas_L.bind_all("<MouseWheel>",self.on_mousewheel)
		side_R=tk.Frame(self.root,bg="#121212",width=300)
		side_R.pack(side="right",fill="y")
		side_R.pack_propagate(False)
		tk.Label(side_R,text="COORDINATE LOC",bg="#121212",fg="#00ff88",font=("Courier",12,"bold")).pack(pady=15)
		self.loc_box=tk.Listbox(side_R,bg="#080808",fg="#00ff88",font=("Courier",11),selectbackground="#004433",bd=0,highlightthickness=0)
		self.loc_box.pack(fill="both",expand=True,padx=10,pady=5)
		self.loc_box.bind("<<ListboxSelect>>",self.on_coord_select)
		work=tk.Frame(self.root,bg="#080808")
		work.pack(side="left",fill="both",expand=True)
		title_f=tk.Frame(work,bg="#080808")
		title_f.pack(pady=10)
		self.global_lbl=tk.Label(title_f,text="GLOBAL: N/A",font=("Courier",16,"bold"),bg="#080808",fg="#00ff88")
		self.global_lbl.pack(side="left",padx=20)
		self.sub_lbl=tk.Label(title_f,text="SUB-IMAGE: N/A",font=("Courier",16,"bold"),bg="#080808",fg="#88ff00")
		self.sub_lbl.pack(side="left",padx=20)
		img_p=tk.Frame(work,bg="#080808")
		img_p.pack()
		self.view_L=tk.Label(img_p,bg="#000",width=420,height=420)
		self.view_L.pack(side="left",padx=15)
		self.view_R=tk.Label(img_p,bg="#000",width=420,height=420)
		self.view_R.pack(side="left",padx=15)
		table_container=tk.Frame(work,bg="#080808")
		table_container.pack(fill="both",expand=True,padx=40,pady=20)
		self.tree_scroll=ttk.Scrollbar(table_container,orient="vertical")
		self.tree=ttk.Treeview(table_container,columns=("T","R","C"),show="headings",yscrollcommand=self.tree_scroll.set)
		self.tree_scroll.config(command=self.tree.yview)
		for c,h in [("T","ANALYSIS LAYER"),("R","DETECTION / FEATURE"),("C","CONFIDENCE / COVERAGE")]:
			self.tree.heading(c,text=h)
			self.tree.column(c,anchor="center")
		self.tree.pack(side="left",fill="both",expand=True)
		self.tree_scroll.pack(side="right",fill="y")
	def on_frame_configure(self,event):
		self.canvas_L.configure(scrollregion=self.canvas_L.bbox("all"))
	def on_mousewheel(self,event):
		self.canvas_L.yview_scroll(int(-1*(event.delta/120)),"units")
	def import_files(self):
		paths=filedialog.askopenfilenames()
		if not paths:
			return
		for w in self.grid_area.winfo_children():
			w.destroy()
		for i,p in enumerate(paths):
			res=analyze(p,grid_size=self.current_grid)
			self.cache[p]=res
			img=ImageTk.PhotoImage(ImageOps.fit(Image.open(p),(100,100)))
			btn=tk.Button(self.grid_area,image=img,bg="#121212",activebackground="#00ff88",command=lambda x=p:self.load_data(x))
			btn.image=img
			btn.grid(row=i//2,column=i%2,padx=10,pady=10)
		self.on_frame_configure(None)
	def load_data(self,p):
		self.active_path=p
		d=self.cache[p]
		self.img_refs['L']=ImageTk.PhotoImage(ImageOps.fit(Image.open(p),(420,420)))
		self.view_L.config(image=self.img_refs['L'])
		self.img_refs['R']=ImageTk.PhotoImage(ImageOps.fit(Image.open(d['seg_path']),(420,420)))
		self.view_R.config(image=self.img_refs['R'])
		self.global_lbl.config(text=f"GLOBAL: {d['global'].upper()}")
		self.loc_box.delete(0,tk.END)
		for item in d['patches']:
			self.loc_box.insert(tk.END,f"  Coord {item['loc']}")
	def on_coord_select(self,event):
		if not self.loc_box.curselection():
			return
		idx=self.loc_box.curselection()[0]
		p=self.cache[self.active_path]['patches'][idx]
		self.sub_lbl.config(text=f"SUB-IMAGE {p['loc']}: {p['vgg_label'].upper()}")
		for i in self.tree.get_children():
			self.tree.delete(i)
		self.tree.insert("", "end",values=("VGG16+AE CLASSIFIER",p['vgg_label'],p['vgg_conf']))
		self.tree.insert("", "end",values=("---","FEATURE BREAKDOWN","---"))
		for f in p['features']:
			tag=f"tag_{f['name'].replace(' ','')}"
			self.tree.insert("", "end",values=("FEATURE",f['name'],f['cov']),tags=(tag,))
			self.tree.tag_configure(tag,foreground=f['hex'])

if __name__=="__main__":
	root=tk.Tk()
	app=SatIntelTerminal(root)
	root.mainloop()