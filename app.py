import os
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template
import timm
import traceback
import sys
from pathlib import Path
import gc

app = Flask(__name__)

# Constants
SR = 16000
CROP = 0.5
N_MELS = 224
FMAX = 8000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
os.makedirs('recordings', exist_ok=True)

# Model Architecture
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

# Initialize model
model = None
model_loaded = False

def load_model():
    global model, model_loaded
    if model_loaded:
        return True
        
    if model is None:
        try:
            print("Starting model loading...")
            print(f"Python version: {sys.version}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            # Memory optimization: set torch threads
            torch.set_num_threads(1)
            
            # Use the available model (from debugging we know it's mobilenetv4_conv_blur_medium)
            model_key = 'mobilenetv4_conv_blur_medium'
            print(f"Using model: {model_key}")
            
            print("Creating model architecture...")
            model = MobileNetV4_Res2TSM(model_key).to(DEVICE)
            
            model_path = "final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth"
            print(f"Loading model weights from: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")
                return False
            
            # Load with memory optimization
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            print("Model state loaded successfully")
            
            if 'state_dict' in state:
                state_dict = state['state_dict']
            elif 'model_state_dict' in state:
                state_dict = state['model_state_dict']
            else:
                state_dict = state
            
            # Remove 'module.' prefix if present
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            print("Loading state dict into model...")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Force garbage collection
            del state_dict, state
            gc.collect()
            
            model_loaded = True
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False
    return True

# Preload model at startup
print("Preloading model at startup...")
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_cough():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'No filename provided'}), 400

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"recordings/cough_{timestamp}.wav"
        
        # Save the file
        audio_file.save(filename)
        
        return jsonify({
            'success': True,
            'filename': filename
        })
    except Exception as e:
        print(f"Error in record_cough: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_coughs():
    try:
        print("Starting cough processing...")
        if not model_loaded:
            print("Model not loaded, attempting to load...")
            if not load_model():
                print("Model loading failed")
                return jsonify({'error': 'Failed to load model'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        files = data.get('files') or data.get('filenames')
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        if len(files) < 5:
            return jsonify({'error': 'At least 5 cough recordings are required'}), 400

        print(f"Processing {len(files)} files...")
        results = []
        
        for file_path in files:
            try:
                print(f"Processing file: {file_path}")
                # Load and process audio
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) == 0:
                    print(f"Empty audio file: {file_path}")
                    continue

                # Trim silence
                y, _ = librosa.effects.trim(y, top_db=20)
                
                # Segment audio
                target_len = int(SR * CROP)
                if len(y) >= target_len:
                    energy = np.convolve(y**2, np.ones(target_len), mode='valid')
                    start = np.argmax(energy)
                    seg = y[start:start+target_len]
                else:
                    seg = np.zeros(target_len, dtype=y.dtype)
                    seg[:len(y)] = y

                # Create mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=seg, sr=SR, n_mels=N_MELS, fmax=FMAX,
                    hop_length=512, win_length=2048
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Resize to target shape
                if mel_db.shape != (224, 224):
                    mel_db = librosa.util.fix_length(mel_db, size=224, axis=1)
                    mel_db = librosa.util.fix_length(mel_db, size=224, axis=0)
                
                # Normalize
                if np.ptp(mel_db) > 0:
                    mel_db = (mel_db - mel_db.min()) / np.ptp(mel_db)
                
                # Convert to RGB
                rgb = np.stack([mel_db] * 3, axis=0)
                
                # Model inference with memory optimization
                tensor = torch.tensor(rgb[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    prob = model(tensor).squeeze().item()
                
                # Clean up tensors
                del tensor, rgb, mel_db, mel
                
                results.append({
                    'filename': os.path.basename(file_path),
                    'probability': prob,
                    'result': 'TB POSITIVE' if prob >= 0.5 else 'TB NEGATIVE',
                    'confidence': prob if prob >= 0.5 else 1 - prob
                })
                print(f"Successfully processed file: {file_path}")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                traceback.print_exc()
                continue

        if not results:
            print("No valid results after processing all files")
            return jsonify({'error': 'Failed to process any files'}), 500

        # Calculate aggregate results
        positive_count = sum(1 for r in results if r['probability'] >= 0.5)
        total_count = len(results)
        avg_prob = np.mean([r['probability'] for r in results])
        
        print(f"Processing complete. Found {total_count} valid results, {positive_count} positive")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            'individual_results': results,
            'summary': {
                'total_coughs': total_count,
                'positive_coughs': positive_count,  
                'negative_coughs': total_count - positive_count,
                'average_probability': avg_prob,
                'overall_result': 'TB POSITIVE' if positive_count >= total_count/2 else 'TB NEGATIVE'
            }
        })

    except Exception as e:
        print(f"Error in process_coughs: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)