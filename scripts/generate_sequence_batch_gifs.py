"""Create compact, explanatory animations for the sequence-learning batch."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = ImageFont.truetype("DejaVuSans.ttf", 22)
TITLE = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
SMALL = ImageFont.truetype("DejaVuSans.ttf", 17)


def frame(title: str, caption: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Return one readable canvas and its drawing context."""
    image = Image.new("RGB", (760, 300), "#101827")
    draw = ImageDraw.Draw(image)
    draw.text((26, 20), title, font=TITLE, fill="#f8fafc")
    draw.text((26, 262), caption, font=SMALL, fill="#cbd5e1")
    return image, draw


def save(slug: str, name: str, frames: list[Image.Image]) -> None:
    """Save a looping GIF under the paper that explains the animation."""
    assets = ROOT / "papers" / slug / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        assets / name,
        save_all=True,
        append_images=frames[1:],
        duration=850,
        loop=0,
        optimize=False,
    )


def lstm() -> list[Image.Image]:
    """Show old cell state surviving when the forget gate is open."""
    frames = []
    for keep in (0.15, 0.55, 0.95):
        image, draw = frame("LSTM: a controlled memory path", f"forget gate = {keep:.2f}")
        draw.rounded_rectangle((55, 125, 245, 195), 14, fill="#164e63")
        draw.text((78, 145), "old cell state", font=FONT, fill="white")
        draw.rounded_rectangle((300, 115, 445, 205), 14, fill="#7c3aed")
        draw.text((326, 135), "forget", font=FONT, fill="white")
        draw.text((326, 164), f"{keep:.2f}", font=FONT, fill="white")
        draw.line((245, 160, 300, 160), fill="#fbbf24", width=7)
        draw.line((445, 160, 625, 160), fill="#fbbf24", width=max(2, int(9 * keep)))
        draw.rounded_rectangle((625, 125, 720, 195), 14, fill="#166534")
        draw.text((640, 145), "next", font=FONT, fill="white")
        frames.append(image)
    return frames


def seq2seq() -> list[Image.Image]:
    """Show encoding followed by one-token-at-a-time decoding."""
    frames = []
    source, targets = ["je", "suis", "ici"], ["i", "am", "here"]
    for step in range(3):
        image, draw = frame("Seq2Seq: summarize, then generate", f"decoder step {step + 1}")
        for index, token in enumerate(source):
            x = 35 + index * 95
            draw.rounded_rectangle((x, 135, x + 75, 185), 10, fill="#0f766e")
            draw.text((x + 15, 149), token, font=SMALL, fill="white")
        draw.polygon([(350, 130), (410, 160), (350, 190)], fill="#fbbf24")
        draw.rounded_rectangle((425, 120, 535, 200), 12, fill="#7c3aed")
        draw.text((443, 143), "LSTM", font=FONT, fill="white")
        for index, token in enumerate(targets):
            x = 570 + index * 58
            color = "#22c55e" if index <= step else "#334155"
            draw.rounded_rectangle((x, 135, x + 48, 185), 9, fill=color)
            draw.text((x + 12, 149), token, font=SMALL, fill="white")
        frames.append(image)
    return frames


def attention() -> list[Image.Image]:
    """Show the decoder focus moving across source positions."""
    frames = []
    source = ["the", "red", "car"]
    for focus in range(3):
        image, draw = frame("Bahdanau attention: retrieve what matters now", f"attention focus: {source[focus]}")
        for index, token in enumerate(source):
            x = 75 + index * 135
            color = "#f59e0b" if index == focus else "#334155"
            draw.rounded_rectangle((x, 120, x + 95, 180), 12, fill=color)
            draw.text((x + 18, 139), token, font=FONT, fill="white")
            draw.line((x + 47, 180, 480, 225), fill=color, width=4 if index == focus else 1)
        draw.rounded_rectangle((425, 205, 610, 250), 12, fill="#7c3aed")
        draw.text((445, 216), "context vector", font=SMALL, fill="white")
        draw.rounded_rectangle((625, 120, 720, 180), 12, fill="#166534")
        draw.text((642, 139), "next", font=FONT, fill="white")
        frames.append(image)
    return frames


def dropout() -> list[Image.Image]:
    """Show independently sampled units disappearing during training."""
    frames = []
    masks = [(1, 0, 1, 0), (0, 1, 1, 1), (1, 1, 0, 1)]
    for mask in masks:
        image, draw = frame("Dropout: train many thinned networks", f"sampled mask: {mask}")
        for index, alive in enumerate(mask):
            y = 70 + index * 50
            color = "#22c55e" if alive else "#475569"
            draw.ellipse((250, y, 286, y + 36), fill=color)
            if alive:
                draw.line((115, 150, 250, y + 18), fill="#38bdf8", width=2)
                draw.line((286, y + 18, 600, 150), fill="#38bdf8", width=2)
        draw.ellipse((85, 132, 125, 172), fill="#f59e0b")
        draw.ellipse((600, 132, 640, 172), fill="#f59e0b")
        draw.text((53, 190), "input", font=SMALL, fill="#cbd5e1")
        draw.text((583, 190), "output", font=SMALL, fill="#cbd5e1")
        frames.append(image)
    return frames


def glove() -> list[Image.Image]:
    """Show count relationships becoming nearby vector locations."""
    frames = []
    points = [("ice", (210, 155)), ("cold", (310, 115)), ("steam", (510, 155)), ("hot", (590, 100))]
    for stage in range(3):
        image, draw = frame("GloVe: counts become geometry", f"optimization step {stage + 1}")
        draw.line((100, 225, 670, 225), fill="#64748b", width=2)
        draw.line((100, 225, 100, 70), fill="#64748b", width=2)
        for token, point in points:
            dx = (stage - 2) * (12 if token in {"ice", "cold"} else -12)
            x, y = point[0] + dx, point[1]
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#f59e0b")
            draw.text((x - 17, y + 15), token, font=SMALL, fill="white")
        draw.text((112, 80), "similar contexts pull related words together", font=SMALL, fill="#cbd5e1")
        frames.append(image)
    return frames


def main() -> None:
    """Generate the five instructional assets."""
    save("31-long-short-term-memory", "cell-state-retention.gif", lstm())
    save("32-sequence-to-sequence-learning", "encoder-decoder.gif", seq2seq())
    save("33-bahdanau-attention", "moving-attention.gif", attention())
    save("34-dropout", "thinned-networks.gif", dropout())
    save("35-glove", "counts-to-vectors.gif", glove())


if __name__ == "__main__":
    main()
