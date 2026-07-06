import cv2
import time
from ultralytics import YOLO

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# OPEN WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

# ==============================
# ALLOWED CLASSES
# ==============================
# person = 0
# car = 2
# bus = 5
# truck = 7
# bottle = 39
# cell phone = 67

allowed_classes = [0, 2, 5, 7, 39, 67]

# ==============================
# COLORS (BGR FORMAT)
# ==============================
colors = {
    0: (255, 0, 0),      # Person → Blue
    2: (0, 0, 255),      # Car → Red
    5: (0, 0, 255),      # Bus → Red
    7: (0, 0, 255),      # Truck → Red
    39: (255, 255, 0),   # Bottle → Cyan
    67: (0, 255, 255)    # Phone → Yellow
}

# ==============================
# FPS VARIABLES
# ==============================
prev_time = 0

# ==============================
# MAIN LOOP
# ==============================
while True:

    # ==========================
    # READ FRAME
    # ==========================
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # ==========================
    # RESIZE FRAME
    # ==========================
    frame = cv2.resize(frame, (640, 480))

    # ==========================
    # FPS CALCULATION
    # ==========================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # ==========================
    # OBJECT COUNTERS
    # ==========================
    person_count = 0
    phone_count = 0
    vehicle_count = 0
    bottle_count = 0

    # ==========================
    # YOLO DETECTION
    # ==========================
    results = model(frame, conf=0.5)

    # ==========================
    # PROCESS DETECTIONS
    # ==========================
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Class ID
            cls = int(box.cls[0])

            # Filter classes
            if cls in allowed_classes:

                # ==================
                # COUNT OBJECTS
                # ==================
                if cls == 0:
                    person_count += 1

                elif cls == 67:
                    phone_count += 1

                elif cls in [2, 5, 7]:
                    vehicle_count += 1

                elif cls == 39:
                    bottle_count += 1

                # ==================
                # BOX COORDINATES
                # ==================
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ==================
                # CONFIDENCE
                # ==================
                confidence = float(box.conf[0])

                # ==================
                # LABEL
                # ==================
                label = model.names[cls]

                text = f"{label}: {confidence:.2f}"

                # ==================
                # COLOR
                # ==================
                color = colors.get(cls, (0, 255, 0))

                # ==================
                # DRAW BOUNDING BOX
                # ==================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # ==================
                # LABEL BACKGROUND
                # ==================
                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - 30),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # ==================
                # DRAW LABEL TEXT
                # ==================
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

    # ==========================
    # DARK FPS PANEL
    # ==========================
    cv2.rectangle(
        frame,
        (10, 10),
        (220, 200),
        (40, 40, 40),
        -1
    )

    # ==========================
    # DISPLAY FPS
    # ==========================
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ==========================
    # DISPLAY COUNTERS
    # ==========================
    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Phones: {phone_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bottles: {bottle_count}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ==========================
    # SHOW WINDOW
    # ==========================
    cv2.imshow("Professional YOLO AI Detection System", frame)

    # ==========================
    # KEYBOARD CONTROLS
    # ==========================
    key = cv2.waitKey(1) & 0xFF

    # SAVE SCREENSHOT
    if key == ord('s'):

        filename = f"screenshots/detection_{int(time.time())}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Screenshot saved: {filename}")

    # QUIT APPLICATION
    if key == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()

cv2.destroyAllWindows()