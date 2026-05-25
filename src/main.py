import sys
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf

MODEL_PATH = "model/mnist_model.keras"


def load_model(path: str = MODEL_PATH):
    """Load the trained model from disk."""
    if not os.path.exists(path):
        print(
            f"Error: model not found at '{path}'.\n"
            "Please run train.py first to train and save the model."
        )
        sys.exit(1)

    model = tf.keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model


def build_gui(model):
    """Build and launch the digit-drawing GUI."""
    root = tk.Tk()
    root.title("MNIST Digit Predictor")
    root.resizable(False, False)

    CANVAS_SIZE = 280   # displayed pixels
    IMG_SIZE    = 28    # model input size
    SCALE       = CANVAS_SIZE // IMG_SIZE

    # PIL image (28×28, grayscale, black background)
    pil_image = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw      = ImageDraw.Draw(pil_image)

    # ── Layout ─────────────────────────────────────────────────────────────────
    frame_top = tk.Frame(root, padx=10, pady=10)
    frame_top.pack()

    canvas = tk.Canvas(
        frame_top,
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        bg="black",
        cursor="crosshair",
    )
    canvas.pack()

    prediction_label = tk.Label(
        root,
        text="Draw a digit above",
        font=("Arial", 14, "bold"),
        pady=8,
    )
    prediction_label.pack()

    confidence_bar_label = tk.Label(root, text="", font=("Arial", 11), pady=2)
    confidence_bar_label.pack()

    reset_button = tk.Button(
        root,
        text="Clear",
        width=12,
        font=("Arial", 11),
        command=lambda: reset_canvas(),
    )
    reset_button.pack(pady=(0, 10))

    # ── Helpers ────────────────────────────────────────────────────────────────
    def reset_canvas():
        canvas.delete("all")
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=0)
        prediction_label.config(text="Draw a digit above")
        confidence_bar_label.config(text="")

    def predict_digit():
        img_array = np.array(pil_image) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        probs          = model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        confidence      = float(np.max(probs)) * 100

        prediction_label.config(
            text=f"Predicted digit: {predicted_digit}   ({confidence:.1f}% confidence)"
        )

        # Simple ASCII confidence bar
        bar_length = int(confidence / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        confidence_bar_label.config(text=bar)

    def on_draw(event):
        x = event.x // SCALE
        y = event.y // SCALE
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            # Draw a small brush (2×2 pixels) so strokes are visible
            draw.ellipse(
                [max(0, x - 1), max(0, y - 1),
                 min(IMG_SIZE - 1, x + 1), min(IMG_SIZE - 1, y + 1)],
                fill=255,
            )
            px, py = x * SCALE, y * SCALE
            r = SCALE + 2
            canvas.create_oval(px - r, py - r, px + r, py + r, fill="white", outline="")
            predict_digit()

    canvas.bind("<B1-Motion>", on_draw)
    canvas.bind("<ButtonPress-1>", on_draw)

    root.mainloop()


def main():
    model = load_model()
    build_gui(model)


if __name__ == "__main__":
    main()
