import os
import subprocess
import tkinter as tk
from tkinter import filedialog
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
    file = filedialog.askopenfilename()
    if file:
        result = analyze(file)
        output_text = (
            f"Result: {result['label']}\n"
            f"Type: {result['domain']}\n"
            f"Confidence: {result['confidence']:.2%}"
        )
        result_label.config(text=output_text, fg="blue")

def launch_gui():
    global result_label
    
    window = tk.Tk()
    window.title("Satellite Image Analyzer")
    window.geometry("450x400")
    
    title = tk.Label(
        window,
        text="Intelligent Satellite Image Analyzer",
        font=("Arial", 16, "bold")
    )
    title.pack(pady=20)
    
    upload_btn = tk.Button(
        window,
        text="Upload & Analyze Image",
        command=upload_image,
        font=("Arial", 12),
        bg="#e1e1e1"
    )
    upload_btn.pack(pady=20)

    result_label = tk.Label(
        window,
        text="Waiting for image...",
        font=("Arial", 12),
        justify="center"
    )
    result_label.pack(pady=20)
    
    window.mainloop()

if __name__ == "__main__":
    check_and_prepare_pipeline()
    launch_gui()
    print("\nPipeline finished successfully.\n")