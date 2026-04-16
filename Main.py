import os
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk
from Analyzer import analyze

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script):
    script_path = os.path.join(BASE_DIR, script)
    if os.path.exists(script_path):
        print(f"\nRunning {script}...\n")
        subprocess.run(["python", script_path])
    else:
        print(f"{script} not found — skipping.")

def check_and_prepare_pipeline():
    print("\nStarting Intelligent Image Analyzer Pipeline...\n")
    if not os.path.exists(os.path.join(BASE_DIR, "models/autoencoder_model.h5")):
        run_script("AutoEncoder.py")
    if not os.path.exists(os.path.join(BASE_DIR, "models/vgg16_model.h5")):
        run_script("VGG16.py")

def upload_image():
    file = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    if file:
        result_label.config(text="Processing image...", foreground="#f39c12")
        window.update_idletasks()
        
        result = analyze(file)
        
        output_text = (
            f"Classification: {result['label']}\n\n"
            f"Domain: {result['domain']}\n\n"
            f"Confidence Score: {result['confidence']:.2%}"
        )
        result_label.config(text=output_text, foreground="#27ae60")

def launch_gui():
    global result_label, window
    
    window = tk.Tk()
    window.title("Satellite Intelligence System")
    window.geometry("500x550")
    window.configure(bg="#f5f6fa")

    # --- Styling ---
    style = ttk.Style()
    style.theme_use('clam')
    
    style.configure("TButton", font=("Segoe UI", 11), padding=10)
    style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), background="#f5f6fa", foreground="#2c3e50")
    style.configure("Card.TFrame", background="white", relief="flat")

    # --- Main Container ---
    main_frame = ttk.Frame(window, padding="30", style="Card.TFrame")
    main_frame.place(relx=0.5, rely=0.5, anchor="center", width=420, height=480)

    # Header
    title = ttk.Label(main_frame, text="Satellite Analyzer", style="Header.TLabel")
    title.pack(pady=(0, 10))

    subtitle = ttk.Label(main_frame, text="Neural Network Terrain Classification", font=("Segoe UI", 9), background="white", foreground="#7f8c8d")
    subtitle.pack(pady=(0, 30))

    # Action Button
    upload_btn = ttk.Button(main_frame, text="SELECT IMAGE", command=upload_image)
    upload_btn.pack(fill="x", pady=10)

    # Divider
    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=20)

    # Result Display Area
    result_label = tk.Label(
        main_frame,
        text="Ready for Analysis",
        font=("Segoe UI", 11),
        bg="white",
        fg="#95a5a6",
        wraplength=350,
        justify="center"
    )
    result_label.pack(expand=True, fill="both")

    # Footer/Exit
    exit_btn = ttk.Button(main_frame, text="Exit Application", command=window.destroy)
    exit_btn.pack(side="bottom", pady=(20, 0))

    window.mainloop()

if __name__ == "__main__":
    check_and_prepare_pipeline()
    launch_gui()