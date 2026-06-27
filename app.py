import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST: Perceptron vs ANN vs CNN",
    page_icon="🧠",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .stApp { background: #0d0f14; color: #e8eaf0; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #12151d 0%, #1a1f2e 50%, #12151d 100%);
    border: 1px solid #2a2f3d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #4f8cff, #a78bfa, #f472b6);
  }
  .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 .5rem; color: #f0f2ff; }
  .hero p  { font-size: 1.05rem; color: #8b90a8; margin: 0; }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
  .metric-card {
    flex: 1; background: #12151d; border: 1px solid #2a2f3d;
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    transition: border-color .2s;
  }
  .metric-card:hover { border-color: #4f8cff; }
  .metric-card .label { font-size: .78rem; color: #8b90a8; letter-spacing: .08em; text-transform: uppercase; }
  .metric-card .value { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono'; margin-top: .3rem; }
  .metric-card.blue  .value { color: #4f8cff; }
  .metric-card.violet .value { color: #a78bfa; }
  .metric-card.green  .value { color: #34d399; }

  /* Section headers */
  .section-header {
    display: flex; align-items: center; gap: .75rem;
    margin: 2rem 0 1rem;
  }
  .section-header .pill {
    background: #1e2230; border: 1px solid #2a2f3d;
    border-radius: 6px; padding: .25rem .7rem;
    font-size: .75rem; font-family: 'JetBrains Mono'; color: #4f8cff;
  }
  .section-header h2 { font-size: 1.3rem; font-weight: 600; margin: 0; color: #e8eaf0; }

  /* Explanation cards */
  .explain-grid { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .explain-card {
    flex: 1; min-width: 220px;
    background: #12151d; border: 1px solid #2a2f3d;
    border-radius: 12px; padding: 1.2rem 1.5rem;
  }
  .explain-card h3 { font-size: 1rem; font-weight: 600; margin: 0 0 .5rem; }
  .explain-card p  { font-size: .88rem; color: #8b90a8; margin: 0; line-height: 1.6; }
  .explain-card.highlight { border-color: #34d399; }
  .explain-card.highlight h3 { color: #34d399; }

  /* Why CNN section */
  .why-cnn {
    background: linear-gradient(135deg, #0f1f18, #12151d);
    border: 1px solid #34d399; border-radius: 14px;
    padding: 1.8rem 2rem; margin: 1.5rem 0;
  }
  .why-cnn h2 { color: #34d399; font-size: 1.3rem; margin: 0 0 1rem; }
  .why-cnn ul { padding-left: 1.2rem; margin: 0; }
  .why-cnn li { color: #c0c6d8; font-size: .92rem; line-height: 1.8; }
  .why-cnn li strong { color: #a7f3d0; }

  /* Matplotlib/Seaborn dark override */
  div[data-testid="stPlotlyChart"], div[data-testid="stImage"] { border-radius: 10px; overflow: hidden; }

  /* Streamlit widget cleanup */
  .stSlider > div { padding: 0; }
  div[data-testid="stExpander"] { background: #12151d; border: 1px solid #2a2f3d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#12151d",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#2a2f3d",
    "axes.labelcolor":  "#8b90a8",
    "xtick.color":      "#8b90a8",
    "ytick.color":      "#8b90a8",
    "text.color":       "#e8eaf0",
    "grid.color":       "#2a2f3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#12151d",
    "legend.edgecolor": "#2a2f3d",
})
COLORS = {"Perceptron": "#4f8cff", "ANN": "#a78bfa", "CNN": "#34d399"}

# ─── Data & Model Loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat

@st.cache_resource(show_spinner=False)
def train_models(epochs):
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_data()

    X_img_train = X_train.reshape(-1, 28, 28)
    X_img_test  = X_test.reshape(-1, 28, 28)
    X_cnn_train = X_train.reshape(-1, 28, 28, 1)
    X_cnn_test  = X_test.reshape(-1, 28, 28, 1)

    # Perceptron
    perceptron = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation="softmax")
    ])
    perceptron.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    h_percp = perceptron.fit(X_img_train, y_train_cat, epochs=epochs, batch_size=32,
                              validation_data=(X_img_test, y_test_cat), verbose=0)

    # ANN
    ann = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h_ann = ann.fit(X_img_train, y_train_cat, epochs=epochs, batch_size=32,
                    validation_data=(X_img_test, y_test_cat), verbose=0)

    # CNN
    cnn = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h_cnn = cnn.fit(X_cnn_train, y_train_cat, epochs=epochs, batch_size=32,
                    validation_data=(X_cnn_test, y_test_cat), verbose=0)

    acc_percp = perceptron.evaluate(X_img_test, y_test_cat, verbose=0)[1]
    acc_ann   = ann.evaluate(X_img_test, y_test_cat, verbose=0)[1]
    acc_cnn   = cnn.evaluate(X_cnn_test,  y_test_cat, verbose=0)[1]

    return (perceptron, ann, cnn,
            h_percp, h_ann, h_cnn,
            acc_percp, acc_ann, acc_cnn,
            X_img_train, X_img_test, X_cnn_train, X_cnn_test)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 MNIST Digit Classification</h1>
  <p>Comparing Perceptron · Artificial Neural Network · Convolutional Neural Network on 70,000 handwritten digits</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training Config")
    epochs = st.slider("Training Epochs", min_value=2, max_value=15, value=5, step=1)
    n_samples = st.slider("Sample images to preview", min_value=3, max_value=10, value=5, step=1)
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.markdown("- **Perceptron** — Flatten → Dense(10)")
    st.markdown("- **ANN** — Flatten → Dense(128) → Dense(64) → Dense(10)")
    st.markdown("- **CNN** — Conv2D×2 → MaxPool×2 → Dense(128) → Dropout → Dense(10)")

# ─── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("Training all three models… this may take a minute ☕"):
    (perceptron, ann, cnn,
     h_percp, h_ann, h_cnn,
     acc_percp, acc_ann, acc_cnn,
     X_img_train, X_img_test,
     X_cnn_train, X_cnn_test) = train_models(epochs)

_, _, y_train, y_test, y_train_cat, y_test_cat = load_data()

# ─── Accuracy Cards ────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">01</span>
  <h2>Test Accuracy Summary</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card blue">
    <div class="label">Perceptron</div>
    <div class="value">{acc_percp*100:.1f}%</div>
  </div>
  <div class="metric-card violet">
    <div class="label">ANN</div>
    <div class="value">{acc_ann*100:.1f}%</div>
  </div>
  <div class="metric-card green">
    <div class="label">CNN</div>
    <div class="value">{acc_cnn*100:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Model Explanations ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">02</span>
  <h2>How Each Model Works</h2>
</div>
<div class="explain-grid">
  <div class="explain-card">
    <h3 style="color:#4f8cff">⚡ Perceptron</h3>
    <p>A single linear layer — the simplest classifier. Flattens each 28×28 image into 784 inputs and maps them directly to 10 class scores. No hidden layers, no non-linearity — limited to learning linear decision boundaries.</p>
  </div>
  <div class="explain-card">
    <h3 style="color:#a78bfa">🔗 ANN (MLP)</h3>
    <p>Adds hidden layers with ReLU activations, letting the network learn non-linear patterns. Still treats the image as a flat vector (784 values), so spatial relationships between pixels are completely lost.</p>
  </div>
  <div class="explain-card highlight">
    <h3>🏆 CNN</h3>
    <p>Preserves the 2D structure of images. Convolutional filters slide across the image to detect edges, curves and shapes locally. Pooling reduces resolution while retaining key features. Dropout fights overfitting. Naturally suited to vision tasks.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Training Curves ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">03</span>
  <h2>Training Curves</h2>
</div>
""", unsafe_allow_html=True)

def plot_history(history, title, color):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#12151d")
    ep = range(1, len(history.history['accuracy']) + 1)
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1f2e")
        ax.spines[:].set_color("#2a2f3d")

    ax1.plot(ep, history.history['accuracy'],     color=color, lw=2, label="Train")
    ax1.plot(ep, history.history['val_accuracy'], color=color, lw=2, ls="--", alpha=.6, label="Val")
    ax1.set_title(f"{title} — Accuracy", color="#e8eaf0", pad=8)
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(ep, history.history['loss'],     color=color, lw=2, label="Train")
    ax2.plot(ep, history.history['val_loss'], color=color, lw=2, ls="--", alpha=.6, label="Val")
    ax2.set_title(f"{title} — Loss", color="#e8eaf0", pad=8)
    ax2.set_xlabel("Epoch"); ax2.legend()

    fig.tight_layout()
    return fig

tab1, tab2, tab3 = st.tabs(["⚡ Perceptron", "🔗 ANN", "🏆 CNN"])
with tab1: st.pyplot(plot_history(h_percp, "Perceptron", COLORS["Perceptron"]))
with tab2: st.pyplot(plot_history(h_ann,   "ANN",        COLORS["ANN"]))
with tab3: st.pyplot(plot_history(h_cnn,   "CNN",        COLORS["CNN"]))

# ─── Validation Accuracy Comparison ───────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">04</span>
  <h2>Validation Accuracy — All Models</h2>
</div>
""", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 4))
ep = range(1, epochs + 1)
ax.plot(ep, h_percp.history['val_accuracy'], color=COLORS["Perceptron"], lw=2.5, marker="o", ms=5, label="Perceptron")
ax.plot(ep, h_ann.history['val_accuracy'],   color=COLORS["ANN"],        lw=2.5, marker="s", ms=5, label="ANN")
ax.plot(ep, h_cnn.history['val_accuracy'],   color=COLORS["CNN"],        lw=2.5, marker="^", ms=5, label="CNN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy")
ax.set_title("Validation Accuracy Comparison", color="#e8eaf0")
ax.legend(); ax.grid(True)
fig.tight_layout()
st.pyplot(fig)

# ─── Sample Predictions ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">05</span>
  <h2>Sample Predictions</h2>
</div>
""", unsafe_allow_html=True)

idxs = np.random.choice(len(X_img_test), n_samples, replace=False)
fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.4, 5))
fig.patch.set_facecolor("#12151d")

for i, idx in enumerate(idxs):
    ax_img = axes[0, i]
    ax_img.imshow(X_img_test[idx].reshape(28, 28), cmap="Blues_r")
    ax_img.set_title(f"True: {y_test[idx]}", color="#e8eaf0", fontsize=9)
    ax_img.axis("off")

    p_pred = np.argmax(perceptron.predict(X_img_test[idx].reshape(1, 28, 28), verbose=0))
    a_pred = np.argmax(ann.predict(X_img_test[idx].reshape(1, 28, 28), verbose=0))
    c_pred = np.argmax(cnn.predict(X_cnn_test[idx].reshape(1, 28, 28, 1), verbose=0))

    ax_pred = axes[1, i]
    ax_pred.axis("off")
    lines = [
        (f"P: {p_pred}", COLORS["Perceptron"] if p_pred == y_test[idx] else "#f87171"),
        (f"A: {a_pred}", COLORS["ANN"]        if a_pred == y_test[idx] else "#f87171"),
        (f"C: {c_pred}", COLORS["CNN"]        if c_pred == y_test[idx] else "#f87171"),
    ]
    for j, (txt, col) in enumerate(lines):
        ax_pred.text(0.5, 1 - j * 0.33, txt, ha="center", va="top",
                     fontsize=9, color=col,
                     transform=ax_pred.transAxes,
                     fontfamily="monospace")

fig.tight_layout(pad=1.2)
st.pyplot(fig)
st.caption("P = Perceptron · A = ANN · C = CNN  |  Green = correct, Red = wrong")

# ─── CNN Confusion Matrix ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">06</span>
  <h2>CNN Confusion Matrix</h2>
</div>
""", unsafe_allow_html=True)

y_pred_cnn = np.argmax(cnn.predict(X_cnn_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred_cnn)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#12151d")
ax.set_facecolor("#1a1f2e")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            linewidths=.5, linecolor="#0d0f14",
            cbar_kws={"shrink": .8},
            ax=ax)
ax.set_title("CNN Confusion Matrix", color="#e8eaf0", pad=12)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
fig.tight_layout()
st.pyplot(fig)

# ─── Why CNN is Best ───────────────────────────────────────────────────────────
st.markdown("""
<div class="why-cnn">
  <h2>🏆 Why CNN Outperforms for Image Tasks</h2>
  <ul>
    <li><strong>Spatial awareness:</strong> Unlike Perceptron and ANN which flatten images to 1D vectors, CNNs preserve the 2D grid structure — pixels retain their neighbourhood relationships.</li>
    <li><strong>Local feature detection:</strong> Convolutional filters act as learned edge/curve/texture detectors, scanning the image locally rather than connecting every pixel to every neuron.</li>
    <li><strong>Parameter efficiency:</strong> A 3×3 filter shared across the entire image has only 9 weights — far fewer than a fully-connected layer over 784 inputs. This reduces overfitting.</li>
    <li><strong>Translation invariance:</strong> Max-pooling makes CNN features robust to small shifts or distortions in digit position — a "7" at the top-left looks like a "7" at the bottom-right.</li>
    <li><strong>Hierarchical learning:</strong> Early layers learn low-level edges; deeper layers combine them into curves, loops, and eventually full digit shapes — mirroring how the visual cortex works.</li>
    <li><strong>Proven scalability:</strong> The same principles that make CNN superior on MNIST (LeNet-5) power modern architectures like ResNet and EfficientNet on millions of images.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ─── Final comparison table (rendered as image) ────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="pill">07</span>
  <h2>Model Comparison Table</h2>
</div>
""", unsafe_allow_html=True)

col_labels  = ["Model", "Test Accuracy", "Spatial Awareness", "Hidden Layers", "Overfitting Risk", "Best For"]
row_data = [
    ["Perceptron",  f"{acc_percp*100:.2f}%", "None",     "0",                  "Low",    "Baselines"],
    ["ANN (MLP)",   f"{acc_ann*100:.2f}%",   "None",     "2 Dense",            "Medium", "Tabular data"],
    ["CNN",         f"{acc_cnn*100:.2f}%",   "Full 2D",  "2 Conv + 1 Dense",   "Low",    "Images / Vision"],
]
row_colors_map = {
    0: ["#1a2035", "#1a2035", "#1a2035", "#1a2035", "#1a2035", "#1a2035"],
    1: ["#1a1f2e", "#1a1f2e", "#1a1f2e", "#1a1f2e", "#1a1f2e", "#1a1f2e"],
    2: ["#0f1f18", "#0f1f18", "#0f1f18", "#0f1f18", "#0f1f18", "#0f1f18"],
}
row_text_colors = {
    0: ["#4f8cff", "#4f8cff", "#f87171", "#e8eaf0", "#e8eaf0", "#e8eaf0"],
    1: ["#a78bfa", "#a78bfa", "#f87171", "#e8eaf0", "#e8eaf0", "#e8eaf0"],
    2: ["#34d399", "#34d399", "#34d399", "#e8eaf0", "#e8eaf0", "#34d399"],
}

fig, ax = plt.subplots(figsize=(12, 2.6))
fig.patch.set_facecolor("#0d0f14")
ax.set_facecolor("#0d0f14")
ax.axis("off")

tbl = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)

# Header style
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("#1e2230")
    cell.set_text_props(color="#8b90a8", fontweight="bold", fontfamily="monospace")
    cell.set_edgecolor("#2a2f3d")

# Data rows
for i, row in enumerate(row_data):
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        cell.set_facecolor(row_colors_map[i][j])
        cell.set_text_props(color=row_text_colors[i][j], fontweight="600" if j == 0 else "normal")
        cell.set_edgecolor("#2a2f3d")

fig.tight_layout(pad=0.5)
st.pyplot(fig)

st.markdown("<br><center style='color:#4a4f62;font-size:.8rem'>Built with Streamlit · TensorFlow/Keras · Scikit-learn</center>", unsafe_allow_html=True)