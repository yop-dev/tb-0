import torch
import torch.nn as nn
import timm

# ======= Model Architecture =======
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
            for _ in range(scale - 1)
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
            sp = self.convs[i - 1](sp)
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

# ======= Load model =======
model = MobileNetV4_Res2TSM('mobilenetv4_conv_blur_medium')

# Load weights safely
state_dict = torch.load('final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth', map_location='cpu')
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
elif 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']

# Handle 'module.' prefix if any
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
model.eval()

# ======= Convert to float16 =======
model.half()

# ======= Save float16 model =======
torch.save(model.state_dict(), 'mobilenetv4_res2tsm_float16.pth')
print("Float16 model saved as 'mobilenetv4_res2tsm_float16.pth'")
