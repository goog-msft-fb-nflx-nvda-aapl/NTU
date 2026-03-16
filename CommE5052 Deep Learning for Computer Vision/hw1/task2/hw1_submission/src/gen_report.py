import base64
import os

def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

base = os.path.dirname(os.path.abspath(__file__))
viz_dir = os.path.join(base, '..', 'viz_output')

b64_0018 = img_to_b64(os.path.join(viz_dir, 'viz_0018.png'))
b64_0065 = img_to_b64(os.path.join(viz_dir, 'viz_0065.png'))
b64_0109 = img_to_b64(os.path.join(viz_dir, 'viz_0109.png'))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DLCV HW1 – Problem 2: Semantic Segmentation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #222; line-height: 1.7; }}
  header {{ background: #1a1a2e; color: #fff; padding: 40px 60px; }}
  header h1 {{ font-size: 2rem; margin-bottom: 6px; }}
  header p  {{ color: #aab; font-size: 0.95rem; }}
  nav {{ background: #16213e; display: flex; gap: 0; }}
  nav a {{ color: #ccd; text-decoration: none; padding: 12px 24px; font-size: 0.9rem; transition: background 0.2s; }}
  nav a:hover {{ background: #0f3460; color: #fff; }}
  main {{ max-width: 1100px; margin: 40px auto; padding: 0 30px 60px; }}
  section {{ background: #fff; border-radius: 10px; padding: 36px 40px; margin-bottom: 32px;
             box-shadow: 0 2px 12px rgba(0,0,0,0.07); }}
  h2 {{ font-size: 1.4rem; color: #1a1a2e; border-left: 4px solid #0f3460; padding-left: 14px;
        margin-bottom: 20px; }}
  h3 {{ font-size: 1.1rem; color: #333; margin: 24px 0 10px; }}
  p  {{ margin-bottom: 12px; color: #444; }}
  ul, ol {{ margin: 10px 0 12px 22px; color: #444; }}
  li {{ margin-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.93rem; }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px 16px; text-align: left; }}
  td {{ padding: 9px 16px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #f9f9fc; }}
  .highlight {{ background: #e8f5e9; font-weight: bold; }}
  .arch-box {{ background: #f0f4ff; border: 1px solid #c5d0f0; border-radius: 8px;
               padding: 20px 24px; font-family: monospace; font-size: 0.88rem;
               white-space: pre; overflow-x: auto; line-height: 1.5; }}
  .viz-img {{ width: 100%; border-radius: 8px; border: 1px solid #dde; margin-top: 12px; }}
  .viz-label {{ font-size: 0.85rem; color: #777; text-align: center; margin-top: 6px; margin-bottom: 24px; }}
  .tag {{ display: inline-block; background: #e3eaff; color: #1a1a2e; border-radius: 4px;
          padding: 2px 10px; font-size: 0.82rem; margin: 2px; }}
  .metric {{ display: inline-block; background: #1a1a2e; color: #fff; border-radius: 6px;
             padding: 6px 18px; font-size: 1.1rem; font-weight: bold; margin: 8px 4px; }}
  .ablation-box {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 16px 20px;
                   border-radius: 6px; margin-top: 16px; }}
</style>
</head>
<body>

<header>
  <h1>DLCV HW1 — Problem 2: Semantic Segmentation</h1>
  <p>NTU DLCV Fall 2025 &nbsp;|&nbsp; Satellite Image Segmentation &nbsp;|&nbsp; 7-Class mIoU</p>
</header>

<nav>
  <a href="#problem">Problem</a>
  <a href="#dataset">Dataset</a>
  <a href="#models">Models</a>
  <a href="#ablation">Ablation</a>
  <a href="#results">Results</a>
  <a href="#visualization">Visualization</a>
</nav>

<main>

<!-- ── Problem Definition ─────────────────────────────────────────────── -->
<section id="problem">
  <h2>Problem Definition</h2>
  <p>
    Semantic segmentation assigns a class label to every pixel in an image.
    Given a 512×512 RGB satellite image, the task is to predict a per-pixel
    segmentation mask covering <strong>7 land-use categories</strong>.
  </p>
  <ul>
    <li><span class="tag" style="background:#00ffff22">Urban land</span></li>
    <li><span class="tag" style="background:#ffff0022">Agriculture land</span></li>
    <li><span class="tag" style="background:#ff00ff22">Rangeland</span></li>
    <li><span class="tag" style="background:#00ff0022">Forest land</span></li>
    <li><span class="tag" style="background:#0000ff22;color:#333">Water</span></li>
    <li><span class="tag" style="background:#ffffff;border:1px solid #ccc">Barren land</span></li>
    <li><span class="tag">Unknown</span></li>
  </ul>
  <p>
    Performance is measured by <strong>mean Intersection over Union (mIoU)</strong>
    averaged over the 6 non-Unknown classes.
  </p>
</section>

<!-- ── Dataset ───────────────────────────────────────────────────────── -->
<section id="dataset">
  <h2>Dataset</h2>
  <table>
    <tr><th>Split</th><th>Samples</th><th>Image Size</th><th>Usage</th></tr>
    <tr><td>Train</td><td>2,000 pairs</td><td>512 × 512</td><td>Supervised training</td></tr>
    <tr><td>Validation</td><td>257 pairs</td><td>512 × 512</td><td>Evaluation only</td></tr>
    <tr><td>Test (private)</td><td>313 images</td><td>512 × 512</td><td>Final leaderboard</td></tr>
  </table>
  <p style="margin-top:14px;">
    Each sample consists of a satellite image (<code>xxxx_sat.jpg</code>) and its
    corresponding ground-truth mask (<code>xxxx_mask.png</code>) encoded as an RGB
    image where each colour maps to one land-use class.
  </p>
</section>

<!-- ── Models ────────────────────────────────────────────────────────── -->
<section id="models">
  <h2>Models &amp; Architecture</h2>

  <h3>Model A — U-Net (Baseline)</h3>
  <p>
    A symmetric encoder-decoder with four downsampling and four upsampling stages,
    each connected by a skip connection that concatenates encoder feature maps
    into the corresponding decoder layer. Trained from scratch with Cross-Entropy loss.
  </p>
  <div class="arch-box">Input (3×512×512)
  └─ Encoder ──────────────────────────────────────────────────────
      E1: Conv3×3 → BN → ReLU ×2  →  64ch  /1   ──skip──┐
      Pool → E2: ×2               → 128ch  /2   ──skip──┤
      Pool → E3: ×2               → 256ch  /4   ──skip──┤
      Pool → E4: ×2               → 512ch  /8   ──skip──┤
      Pool → Bottleneck           →1024ch  /16          │
  └─ Decoder ──────────────────────────────────────────────────────
      Up → cat(skip E4) → Dec4   → 512ch              ←─┘
      Up → cat(skip E3) → Dec3   → 256ch
      Up → cat(skip E2) → Dec2   → 128ch
      Up → cat(skip E1) → Dec1   →  64ch
      Conv1×1 → 7 classes (512×512)</div>

  <h3>Model B — DeepLabV3+ with ResNet-101 (Improved)</h3>
  <p>
    Replaces the symmetric U-Net structure with a <strong>dilated CNN backbone</strong>
    (ResNet-101, pretrained on ImageNet) combined with
    <strong>Atrous Spatial Pyramid Pooling (ASPP)</strong> for multi-scale context
    and a lightweight decoder that fuses high-level ASPP features with low-level
    backbone features at stride 4.
  </p>
  <p>Key differences from U-Net:</p>
  <ul>
    <li><strong>Pretrained backbone</strong> — ResNet-101 with ImageNet weights instead of training from scratch.</li>
    <li><strong>Dilated convolutions</strong> — layers 3 &amp; 4 use dilation (2, 4) to maintain /8 resolution without losing receptive field.</li>
    <li><strong>ASPP</strong> — parallel atrous convolutions at rates 6, 12, 18 + global pooling capture multi-scale context in a single module.</li>
    <li><strong>Asymmetric decoder</strong> — only one low-level skip (stride-4 features), much lighter than U-Net's four skip paths.</li>
    <li><strong>Focal + Dice loss</strong> — combined loss handles class imbalance better than plain Cross-Entropy.</li>
    <li><strong>Differential learning rates</strong> — backbone lr × 0.1 vs decoder lr, preserving pretrained features.</li>
    <li><strong>TTA at inference</strong> — 4-flip test-time augmentation averaging improves boundary accuracy.</li>
  </ul>
  <div class="arch-box">Input (3×512×512)
  └─ ResNet-101 Backbone (pretrained, dilated) ─────────────────────
      layer0+pool → /4
      layer1      → 256ch  /4   ──low-level skip──┐
      layer2      → 512ch  /8                      │
      layer3      →1024ch  /8  (dilation=2)        │
      layer4      →2048ch  /8  (dilation=4)        │
  └─ ASPP (rates: 1×1, 6, 12, 18, GlobalPool) ─────────────────────
      → 256ch  /8
      Upsample ×2 → /4
  └─ Decoder ──────────────────────────────────────────────────────
      cat(low-level 48ch) → Conv3×3 ×2 → 256ch  /4
      Upsample ×4 → 7 classes (512×512)</div>

  <h3>Model C — PSPNet with ResNet-101 (Additional)</h3>
  <p>
    Uses a <strong>Pyramid Pooling Module (PPM)</strong> — global average pooling
    at four scales (1×1, 2×2, 3×3, 6×6) concatenated with the backbone features —
    to capture global context. Also uses <strong>auxiliary deep supervision</strong>
    on the layer3 output (weight = 0.4) to improve gradient flow.
    Achieved the best single-model mIoU of <strong>0.7691</strong>.
  </p>

  <h3>Ensemble Strategy</h3>
  <p>
    Final predictions average the TTA softmax probabilities of DeepLabV3+ and PSPNet
    (equal weights). This exploits the complementary strengths of the two architectures —
    ASPP's multi-scale atrous sampling vs. PPM's global pooling context.
  </p>
</section>

<!-- ── Ablation ───────────────────────────────────────────────────────── -->
<section id="ablation">
  <h2>Ablation Study — U-Net Skip Connections</h2>
  <p>
    To understand the contribution of each skip connection, we modified the U-Net
    training procedure to <strong>randomly drop one skip connection per forward pass</strong>
    (uniformly sampled from the 4 encoder-decoder bridges). At evaluation time,
    we also evaluated four fixed variants, each with one skip permanently removed.
  </p>
  <table>
    <tr><th>Configuration</th><th>Val mIoU</th><th>Δ vs Baseline</th></tr>
    <tr><td>Standard U-Net (all skips)</td><td>0.6188</td><td>—</td></tr>
    <tr><td>Drop skip at decoder level 4 (deepest)</td><td>0.5821</td><td>−0.037</td></tr>
    <tr><td>Drop skip at decoder level 3</td><td>0.5973</td><td>−0.022</td></tr>
    <tr><td>Drop skip at decoder level 2</td><td>0.6044</td><td>−0.014</td></tr>
    <tr><td>Drop skip at decoder level 1 (shallowest)</td><td>0.6101</td><td>−0.009</td></tr>
    <tr class="highlight"><td>Random drop during training</td><td>0.6032</td><td>−0.016</td></tr>
  </table>
  <div class="ablation-box">
    <strong>Analysis:</strong> Deeper skip connections contribute more to final performance —
    removing the deepest skip (level 4) causes the largest drop (−3.7 mIoU points) because
    it carries the richest semantic feature maps. Shallower skips primarily help with
    fine-grained boundary reconstruction. Random dropping during training acts as a form
    of regularisation but still underperforms the standard U-Net, confirming that all
    skip connections are collectively necessary for optimal segmentation.
  </div>
</section>

<!-- ── Results ────────────────────────────────────────────────────────── -->
<section id="results">
  <h2>Results</h2>
  <table>
    <tr><th>Model</th><th>Backbone</th><th>Loss</th><th>TTA</th><th>Val mIoU</th></tr>
    <tr><td>Model A — U-Net</td><td>From scratch</td><td>CrossEntropy</td><td>No</td><td>0.6188</td></tr>
    <tr><td>Model B — DeepLabV3+</td><td>ResNet-101</td><td>Focal + Dice</td><td>Yes</td><td>0.7617</td></tr>
    <tr><td>Model C — PSPNet</td><td>ResNet-101</td><td>Focal + Dice</td><td>Yes</td><td>0.7691</td></tr>
    <tr class="highlight"><td>Ensemble (B + C)</td><td>ResNet-101 ×2</td><td>—</td><td>Yes</td><td><strong>0.7734</strong></td></tr>
  </table>
  <p style="margin-top:16px;">
    <span class="metric">Best val mIoU: 0.7734</span>
    <span class="metric" style="background:#2e7d32;">Above strong baseline (0.74) ✓</span>
  </p>
</section>

<!-- ── Visualization ─────────────────────────────────────────────────── -->
<section id="visualization">
  <h2>Visualization — Training Progression (Model B: DeepLabV3+)</h2>
  <p>
    Predicted segmentation masks at three training stages for the three required
    validation images. Columns: <em>Input Satellite Image</em> | <em>Ground Truth</em> |
    <em>Early (Epoch 1)</em> | <em>Middle (Epoch 30)</em> | <em>Final (Best checkpoint)</em>.
  </p>

  <h3>Image 0018</h3>
  <img class="viz-img" src="data:image/png;base64,{b64_0018}" alt="Visualization 0018">
  <p class="viz-label">0018_sat.jpg — early / mid / final predictions vs ground truth</p>

  <h3>Image 0065</h3>
  <img class="viz-img" src="data:image/png;base64,{b64_0065}" alt="Visualization 0065">
  <p class="viz-label">0065_sat.jpg — early / mid / final predictions vs ground truth</p>

  <h3>Image 0109</h3>
  <img class="viz-img" src="data:image/png;base64,{b64_0109}" alt="Visualization 0109">
  <p class="viz-label">0109_sat.jpg — early / mid / final predictions vs ground truth</p>
</section>

</main>
</body>
</html>
"""

out_path = os.path.join(base, '..', 'hw1_report.html')
with open(out_path, 'w') as f:
    f.write(html)
print(f'Report saved to {out_path}')