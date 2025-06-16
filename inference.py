# tb_detection_gui.py

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix NumPy compatibility issue
import numpy as np
if hasattr(np, '__version__') and tuple(map(int, np.__version__.split('.')[:2])) >= (2, 0):
    print("Warning: NumPy 2.x detected. Consider downgrading for better compatibility.")

import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import time

# Try importing librosa with fallback
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except Exception as e:
    print(f"Warning: librosa import failed: {e}")
    LIBROSA_AVAILABLE = False

# Try scipy import with fallback
try:
    from scipy.ndimage import zoom
    SCIPY_ZOOM_AVAILABLE = True
except Exception as e:
    print(f"Warning: scipy.ndimage.zoom import failed: {e}")  
    SCIPY_ZOOM_AVAILABLE = False

import timm 

# ─── Constants ───────────────────────────────────────────────────────────────
SR = 16000
CROP = 0.5
N_MELS = 224
FMAX = 8000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alternative implementations for compatibility
def simple_zoom(array, zoom_factors, order=1):
    """Simple zoom implementation using numpy only"""
    if not SCIPY_ZOOM_AVAILABLE:
        from numpy import interp
        old_shape = array.shape
        new_shape = tuple(int(old_shape[i] * zoom_factors[i]) for i in range(len(old_shape)))
        
        if len(array.shape) == 2:
            # 2D case - simple nearest neighbor interpolation
            result = np.zeros(new_shape)
            for i in range(new_shape[0]):
                for j in range(new_shape[1]):
                    old_i = int(i * old_shape[0] / new_shape[0])
                    old_j = int(j * old_shape[1] / new_shape[1])
                    old_i = min(old_i, old_shape[0] - 1)
                    old_j = min(old_j, old_shape[1] - 1)
                    result[i, j] = array[old_i, old_j]
            return result
        else:
            return array
    else:
        return zoom(array, zoom_factors, order=order)

def load_audio_fallback(path, sr=16000):
    """Fallback audio loading without librosa"""
    try:
        import soundfile as sf
        y, orig_sr = sf.read(path)
        if orig_sr != sr:
            ratio = sr / orig_sr
            new_length = int(len(y) * ratio)
            y = np.interp(np.linspace(0, len(y), new_length), np.arange(len(y)), y)
        return y
    except ImportError:
        try:
            import wave
            with wave.open(path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return y
        except Exception as e:
            return None

# ─── Model Components ────────────────────────────────────────────────────────
class TemporalShift(nn.Module):
    def __init__(self, channels, shift_div=8):
        super().__init__()
        self.fold = channels // shift_div

    def forward(self, x):
        B, C, T, F = x.size()
        t = x.permute(0, 2, 1, 3).contiguous()
        out = torch.zeros_like(t)
        out[:, :-1, :self.fold, :] = t[:, 1:, :self.fold, :]
        out[:, 1:, self.fold:2*self.fold, :] = t[:, :-1, self.fold:2*self.fold, :]
        out[:, :, 2*self.fold:, :] = t[:, :, 2*self.fold:, :]
        return out.permute(0, 2, 1, 3)

class Res2TSMBlock(nn.Module):
    def __init__(self, channels, scale=4, shift_div=8):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale
        self.temporal_shift = TemporalShift(channels, shift_div)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.temporal_shift(x)
        splits = torch.split(x, self.width, dim=1)
        y = splits[0]
        outs = [y]
        for i in range(1, self.scale):
            sp = splits[i] + y
            sp = self.convs[i-1](sp)
            y = sp
            outs.append(sp)
        out = torch.cat(outs, dim=1)
        out = self.bn(out)
        return self.act(out)

class MobileNetV4_Res2TSM(nn.Module):
    def __init__(self, model_key, scale=4, shift_div=8, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_key, pretrained=False, features_only=True)
        C = self.backbone.feature_info.channels()[-1]
        self.res2tsm = Res2TSMBlock(C, scale=scale, shift_div=shift_div)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.res2tsm(feat)
        out = self.global_pool(feat).view(feat.size(0), -1)
        return self.fc(out).squeeze(1)

# ─── Audio Processing Functions ──────────────────────────────────────────────
def load_and_segment(path):
    try:
        if LIBROSA_AVAILABLE:
            try:
                y, _ = librosa.load(path, sr=SR)
            except Exception:
                y = load_audio_fallback(path, sr=SR)
                if y is None:
                    return None
        else:
            y = load_audio_fallback(path, sr=SR)
            if y is None:
                return None
        
        if len(y) == 0:
            return None
        
        # Simple silence trimming
        if LIBROSA_AVAILABLE:
            try:
                y, _ = librosa.effects.trim(y, top_db=20)
            except:
                energy = y**2
                threshold = np.max(energy) * 0.01
                valid_indices = np.where(energy > threshold)[0]
                if len(valid_indices) > 0:
                    y = y[valid_indices[0]:valid_indices[-1]+1]
        else:
            energy = y**2
            threshold = np.max(energy) * 0.01
            valid_indices = np.where(energy > threshold)[0]
            if len(valid_indices) > 0:
                y = y[valid_indices[0]:valid_indices[-1]+1]
        
        target_len = int(SR * CROP)
        if len(y) >= target_len:
            energy = np.convolve(y**2, np.ones(target_len), mode='valid')
            start = np.argmax(energy)
            seg = y[start:start+target_len]
        else:
            seg = np.zeros(target_len, dtype=y.dtype)
            seg[:len(y)] = y
            
        return seg
    except Exception:
        return None

def make_mel_rgb(y_seg):
    try:
        if LIBROSA_AVAILABLE:
            try:
                mel = librosa.feature.melspectrogram(
                    y=y_seg, sr=SR, n_mels=N_MELS, fmax=FMAX,
                    hop_length=512, win_length=2048
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
            except Exception:
                # Simple fallback spectrogram
                n_fft = 2048
                hop_length = 512
                stft = np.abs(np.fft.fft(y_seg.reshape(-1, n_fft), axis=1))
                mel_db = 10 * np.log10(stft[:N_MELS, :224] + 1e-10)
        else:
            # Very basic spectrogram
            n_fft = 2048
            stft = np.abs(np.fft.fft(y_seg.reshape(-1, n_fft), axis=1))
            mel_db = 10 * np.log10(stft[:N_MELS, :224] + 1e-10)
        
        target_shape = (224, 224)
        if mel_db.shape != target_shape:
            zoom_factors = (target_shape[0]/mel_db.shape[0], target_shape[1]/mel_db.shape[1])
            resized = simple_zoom(mel_db, zoom_factors, order=1)
        else:
            resized = mel_db
        
        if np.ptp(resized) > 0:
            normed = (resized - resized.min()) / np.ptp(resized)
        else:
            normed = np.zeros_like(resized)
            
        rgb = np.stack([normed] * 3, axis=0)
        return resized, rgb
    except Exception:
        return None, None

# ─── GUI Application ─────────────────────────────────────────────────────────
class TBDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TB Cough Detection System")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Model and data
        self.model = None
        self.audio_files = []
        self.results = []
        
        # Threading
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="TB Cough Detection System", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Controls section
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        control_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Button(control_frame, text="Select Audio Files", 
                  command=self.select_files, width=20).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        self.file_count_label = ttk.Label(control_frame, text="No files selected")
        self.file_count_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0), pady=5)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Process Files", 
                                     command=self.process_files, state=tk.DISABLED, width=20)
        self.process_btn.grid(row=1, column=0, pady=10, sticky=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=1, column=1, pady=10, padx=(20, 0), sticky=(tk.W, tk.E))
        
        # Status and model info
        info_frame = ttk.Frame(control_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(info_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.model_status_label = ttk.Label(info_frame, text="Loading model...", 
                                           foreground="orange")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(30, 0))
        
        device_label = ttk.Label(info_frame, text=f"Device: {DEVICE}")
        device_label.grid(row=0, column=2, sticky=tk.W, padx=(30, 0))
        
        # MAIN RESULT DISPLAY - Very prominent
        result_frame = ttk.LabelFrame(main_frame, text="DETECTION RESULT", padding="20")
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        result_frame.columnconfigure(0, weight=1)
        
        # Large, prominent result display
        self.main_result_label = ttk.Label(result_frame, text="NO ANALYSIS PERFORMED", 
                                          font=('Arial', 24, 'bold'), foreground="gray")
        self.main_result_label.grid(row=0, column=0, pady=(10, 5))
        
        # Confidence/summary info
        self.confidence_label = ttk.Label(result_frame, text="", 
                                         font=('Arial', 14), foreground="gray")
        self.confidence_label.grid(row=1, column=0, pady=(0, 10))
        
        # Detailed results section
        details_frame = ttk.LabelFrame(main_frame, text="Detailed Results", padding="10")
        details_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # Results table
        columns = ('File', 'Result', 'Confidence', 'Probability')
        self.results_tree = ttk.Treeview(details_frame, columns=columns, show='headings', height=12)
        
        # Define headings
        self.results_tree.heading('File', text='Filename')
        self.results_tree.heading('Result', text='Result')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.heading('Probability', text='Probability')
        
        # Define column widths
        self.results_tree.column('File', width=250)
        self.results_tree.column('Result', width=120)
        self.results_tree.column('Confidence', width=100)
        self.results_tree.column('Probability', width=100)
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, 
                                        command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Export button
        self.export_btn = ttk.Button(details_frame, text="Export Results", 
                                    command=self.export_results, state=tk.DISABLED)
        self.export_btn.grid(row=1, column=0, pady=(10, 0))
        
        # Log panel
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_model(self):
        """Load the ML model in a separate thread"""
        def load_model_thread():
            try:
                self.log_message("Loading TB detection model...")
                model = MobileNetV4_Res2TSM('mobilenetv4_conv_blur_medium').to(DEVICE)
                
                model_path = "final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                state = torch.load(model_path, map_location=DEVICE)
                
                if 'state_dict' in state:
                    state_dict = state['state_dict']
                elif 'model_state_dict' in state:
                    state_dict = state['model_state_dict']
                else:
                    state_dict = state
                
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                self.model = model
                self.root.after(0, lambda: [
                    self.model_status_label.configure(text="Model loaded successfully", foreground="green"),
                    self.log_message("Model loaded successfully!")
                ])
                
            except Exception as e:
                self.root.after(0, lambda: [
                    self.model_status_label.configure(text="Model load failed", foreground="red"),
                    self.log_message(f"Error loading model: {str(e)}"),
                    messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
                ])
        
        threading.Thread(target=load_model_thread, daemon=True).start()
        
    def select_files(self):
        """Select audio files for processing"""
        files = filedialog.askopenfilenames(
            title="Select cough audio files",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.audio_files = list(files)
            self.file_count_label.configure(text=f"{len(files)} files selected")
            self.process_btn.configure(state=tk.NORMAL if self.model else tk.DISABLED)
            self.log_message(f"Selected {len(files)} audio files")
            
            # Clear previous results
            self.clear_results()
        
    def clear_results(self):
        """Clear previous results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results = []
        self.main_result_label.configure(text="NO ANALYSIS PERFORMED", foreground="gray")
        self.confidence_label.configure(text="", foreground="gray")
        self.export_btn.configure(state=tk.DISABLED)
        
    def process_files(self):
        """Process selected audio files"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded yet!")
            return
            
        if not self.audio_files:
            messagebox.showerror("Error", "No files selected!")
            return
        
        self.process_btn.configure(state=tk.DISABLED)
        self.is_processing = True
        self.status_label.configure(text="Processing...", foreground="blue")
        self.progress_var.set(0)
        
        # Start processing in separate thread
        threading.Thread(target=self.process_files_thread, daemon=True).start()
        
    def process_files_thread(self):
        """Process files in background thread"""
        try:
            results = []
            
            total_files = len(self.audio_files)
            self.log_message(f"Starting processing of {total_files} files...")
            
            for i, file_path in enumerate(self.audio_files):
                filename = os.path.basename(file_path)
                self.root.after(0, lambda f=filename: self.log_message(f"Processing: {f}"))
                
                # Process audio
                seg = load_and_segment(file_path)
                if seg is None:
                    self.root.after(0, lambda f=filename: self.log_message(f"Failed to load: {f}"))
                    continue
                
                mel_db, rgb = make_mel_rgb(seg)
                if mel_db is None or rgb is None:
                    self.root.after(0, lambda f=filename: self.log_message(f"Failed to create spectrogram: {f}"))
                    continue
                
                # Model inference
                tensor = torch.tensor(rgb[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    prob = self.model(tensor).item()
                
                result = {
                    'filename': filename,
                    'probability': prob,
                    'result': 'TB POSITIVE' if prob >= 0.5 else 'TB NEGATIVE',
                    'confidence': prob if prob >= 0.5 else 1 - prob
                }
                
                results.append(result)
                
                # Update progress
                progress = (i + 1) / total_files * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Update UI with results
            self.root.after(0, lambda: self.update_results(results))
            
        except Exception as e:
            self.root.after(0, lambda: [
                self.log_message(f"Processing error: {str(e)}"),
                messagebox.showerror("Processing Error", f"An error occurred:\n{str(e)}"),
                self.reset_processing_state()
            ])
    
    def update_results(self, results):
        """Update UI with processing results"""
        try:
            self.results = results
            
            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Add new results
            for result in results:
                self.results_tree.insert('', tk.END, values=(
                    result['filename'],
                    result['result'],
                    f"{result['confidence']*100:.1f}%",
                    f"{result['probability']:.4f}"
                ))
            
            # Update main result display
            if results:
                positive_count = sum(1 for r in results if r['probability'] >= 0.5)
                total_count = len(results)
                avg_prob = np.mean([r['probability'] for r in results])
                
                # Determine overall result (majority vote)
                positive_ratio = positive_count / total_count
                overall_positive = positive_ratio >= 0.5
                
                # Update main result display with clear, prominent text
                if overall_positive:
                    main_text = "🔴 TB POSITIVE DETECTED"
                    main_color = "red"
                    confidence_text = f"{positive_count} of {total_count} files positive ({positive_ratio*100:.1f}%)\nAverage probability: {avg_prob:.3f}"
                else:
                    main_text = "✅ TB NEGATIVE"
                    main_color = "green"
                    negative_ratio = (total_count - positive_count) / total_count
                    confidence_text = f"{total_count - positive_count} of {total_count} files negative ({negative_ratio*100:.1f}%)\nAverage probability: {avg_prob:.3f}"
                
                self.main_result_label.configure(text=main_text, foreground=main_color)
                self.confidence_label.configure(text=confidence_text, foreground=main_color)
                
                self.export_btn.configure(state=tk.NORMAL)
                self.log_message(f"ANALYSIS COMPLETE - Overall Result: {'TB POSITIVE' if overall_positive else 'TB NEGATIVE'} ({positive_count}/{total_count} files positive)")
            
            self.reset_processing_state()
            
        except Exception as e:
            self.log_message(f"Error updating results: {str(e)}")
            self.reset_processing_state()
    
    def reset_processing_state(self):
        """Reset processing state"""
        self.is_processing = False
        self.process_btn.configure(state=tk.NORMAL if self.model and self.audio_files else tk.DISABLED)
        self.status_label.configure(text="Ready", foreground="green")
        self.progress_var.set(100)
    
    def export_results(self):
        """Export results to CSV file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save results as..."
            )
            
            if filename:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'result', 'probability', 'confidence'])
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(result)
                
                # Add summary
                positive_count = sum(1 for r in self.results if r['probability'] >= 0.5)
                total_count = len(self.results)
                avg_prob = np.mean([r['probability'] for r in self.results])
                overall_result = "TB POSITIVE" if positive_count >= total_count/2 else "TB NEGATIVE"
                
                with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                    csvfile.write(f"\n\nSUMMARY\n")
                    csvfile.write(f"Overall Result,{overall_result}\n")
                    csvfile.write(f"Total Files,{total_count}\n")
                    csvfile.write(f"TB Positive Files,{positive_count}\n")
                    csvfile.write(f"TB Negative Files,{total_count - positive_count}\n")
                    csvfile.write(f"Average Probability,{avg_prob:.4f}\n")
                
                messagebox.showinfo("Success", f"Results exported successfully to:\n{filename}")
                self.log_message(f"Results exported to: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
            self.log_message(f"Export error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TBDetectionGUI(root)
    root.mainloop()