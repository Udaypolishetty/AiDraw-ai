"""
AirDraw AI — Phase 2/3 canvas + Phase 4/5/6 recognition + Phase 7 image result,
all combined into a single window (camera+canvas on the left, result on the right).
"""
import cv2
import numpy as np

import config
from src.hand_tracker import HandTracker
from src.air_canvas import AirCanvas
from src.input_classifier import InputClassifier
from src.image_provider import FallbackImageProvider, ImageGenerationProvider, LocalImageProvider


def draw_landmarks(frame, hand):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for start, end in connections:
        cv2.line(frame, hand.points[start], hand.points[end], (0, 150, 0), 1)
    tip = hand.points[config.INDEX_FINGER_TIP]
    cv2.circle(frame, tip, 8, (0, 0, 255), -1)


def draw_result_banner(frame, result):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 90), (w, h), (30, 30, 30), -1)

    if result is None:
        cv2.putText(frame, "Press 's' to recognize your drawing", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return

    if result.input_type == "UNKNOWN":
        text = "Not confident enough to recognize that — try again"
        color = (0, 0, 255)
    else:
        text = f"[{result.input_type}] {result.label}  ({result.confidence:.1f}% confidence)"
        color = (0, 255, 255)
    cv2.putText(frame, text, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    detail = (f"OCR: {result.ocr_result.text or '-'} ({result.ocr_result.confidence:.1f}%)   "
              f"Drawing: {result.drawing_result.label} ({result.drawing_result.confidence:.1f}%)")
    cv2.putText(frame, detail, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def build_result_panel(label, image, height, width, is_loading=False):
    """
    Builds the right-hand 'Generated Image' panel, sized to match the
    main frame's height so it can sit side by side in one window.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    if is_loading:
        cv2.putText(panel, "Generating image...", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        return panel

    if image is not None:
        img_area_size = min(width, height - 60)
        resized = cv2.resize(image, (img_area_size, img_area_size))
        y_offset = 60
        x_offset = (width - img_area_size) // 2
        panel[y_offset:y_offset + img_area_size, x_offset:x_offset + img_area_size] = resized
    elif label:
        cv2.putText(panel, "No image available", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    else:
        cv2.putText(panel, "Draw something and", (20, height // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1)
        cv2.putText(panel, "press 's' to see it here", (20, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1)

    if label:
        cv2.putText(panel, f"Result: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    return panel


def combine_views(main_frame, result_panel):
    divider = np.full((main_frame.shape[0], 4, 3), (60, 60, 60), dtype=np.uint8)
    return np.hstack([main_frame, divider, result_panel])


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check CAMERA_INDEX in config.py.")
        return

    try:
        tracker = HandTracker()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        cap.release()
        return

    canvas = AirCanvas()

    print("Loading recognition models (first run downloads them, be patient)...")
    classifier = InputClassifier()
    image_provider = FallbackImageProvider([
        ImageGenerationProvider(),
        LocalImageProvider(),
    ])
    print("Models loaded.")

    last_result = None
    last_image = None
    result_panel_width = 400

    print("AirDraw AI - Phase 7: Image Result")
    print("Point with index finger (other fingers curled) to draw.")
    print("Keys: c = clear | z = undo | s = recognize + generate image | q = quit")

    while True:
        success, frame = cap.read()
        if not success:
            print("WARNING: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        hand = tracker.process(frame)
        pen_down = False
        if hand is not None:
            draw_landmarks(frame, hand)
            fingertip = tracker.get_index_fingertip(hand)
            pen_down = tracker.is_index_finger_up(hand)
            canvas.update(fingertip, pen_down)
        else:
            canvas.update((0, 0), pen_down=False)

        display_frame = canvas.render_overlay(frame)

        status = "DRAWING" if pen_down else ("no hand" if hand is None else "pen up")
        color = (0, 255, 255) if pen_down else (0, 0, 255) if hand is None else (150, 150, 150)
        cv2.putText(display_frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        draw_result_banner(display_frame, last_result)

        result_panel = build_result_panel(
            last_result.label if last_result else None,
            last_image,
            display_frame.shape[0],
            result_panel_width,
        )
        combined = combine_views(display_frame, result_panel)
        cv2.imshow("AirDraw AI", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()
            last_result = None
            last_image = None
        elif key == ord('z'):
            canvas.undo()
        elif key == ord('s'):
            if not canvas.has_content():
                print("Nothing drawn yet.")
                continue

            print("Recognizing...")
            last_result = classifier.classify(canvas.get_canvas())
            print(f"-> {last_result.input_type}: {last_result.label} "
                  f"({last_result.confidence:.1f}%)")

            if last_result.input_type == "UNKNOWN":
                last_image = None
                continue

            # Show a "Generating..." state immediately so the window doesn't
            # look frozen while the network request runs.
            loading_panel = build_result_panel(last_result.label, None,
                                                display_frame.shape[0], result_panel_width,
                                                is_loading=True)
            cv2.imshow("AirDraw AI", combine_views(display_frame, loading_panel))
            cv2.waitKey(1)

            print(f"Fetching image for '{last_result.label}'... (this can take a few seconds)")
            last_image = image_provider.get_image(last_result.label)

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()