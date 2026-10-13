
import cv2
import numpy as np

VIDEO_PATH = "GX010285.MP4"
WIDTH, HEIGHT = 640, 360

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open ants.mp4")

# Initial MOG2 settings
bg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# Create windows and sliders
cv2.namedWindow("Controls")
cv2.namedWindow("Ant Detection")

def save_parameters():
    parameters = {
        "varThreshold": var_threshold,
        "kernel_size": kernel_size,
        "min_area": min_area,
        "learning_rate": learning_rate
    }

    with open("ant_parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)

    print("Parameters saved!")

def nothing(value):
    pass

cv2.createTrackbar("Var Threshold", "Controls", 16, 100, nothing)
cv2.createTrackbar("Kernel Size", "Controls", 3, 15, nothing)
cv2.createTrackbar("Min Area", "Controls", 10, 500, nothing)

# 0 means automatic learning rate.
# Other values represent learning rate / 10000.
cv2.createTrackbar("Learning Rate x10000", "Controls", 0, 100, nothing)

paused = False
last_frame = None
last_foreground = None

while True:

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        last_frame = frame.copy()

    frame = last_frame.copy()

    # Read slider settings
    var_threshold = max(
        1, cv2.getTrackbarPos("Var Threshold", "Controls")
    )

    kernel_size = max(
        1, cv2.getTrackbarPos("Kernel Size", "Controls")
    )

    # Kernel sizes should be odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    min_area = cv2.getTrackbarPos(
        "Min Area", "Controls"
    )

    lr_value = cv2.getTrackbarPos(
        "Learning Rate x10000", "Controls"
    )

    learning_rate = (
        -1 if lr_value == 0
        else lr_value / 10000.0
    )

    # Update the MOG2 threshold without recreating
    # the model (which would lose its learned background).
    bg.setVarThreshold(var_threshold)

    if not paused:
        # Update the model once for each new video frame.
        last_foreground = bg.apply(
            frame,
            learningRate=learning_rate
        )

    foreground = last_foreground.copy()

    # Only foreground pixels, excluding detected shadows
    binary = cv2.inRange(foreground, 255, 255)

    # Create kernel using the slider value
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # Remove small noise
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    # Fill small holes
    cleaned = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel
    )

    # Connected component analysis
    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
    )

    ant_candidates = 0

    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        ant_candidates += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            frame,
            str(area),
            (x, max(15, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1
        )

    # Show live parameter values
    info = [
        f"varThreshold: {var_threshold}",
        f"Kernel: {kernel_size}x{kernel_size}",
        f"Minimum area: {min_area}",
        f"Learning rate: {learning_rate:.4f}",
        f"Candidates: {ant_candidates}"
    ]

    # Black rectangle behind text
    cv2.rectangle(frame, (0, 0), (270, 125), (0, 0, 0), -1)

    for i, line in enumerate(info):
        cv2.putText(
            frame,
            line,
            (10, 20 + 23 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    cv2.imshow("Ant Detection", frame)
    cv2.imshow("Original Foreground Mask", binary)
    cv2.imshow("Cleaned Mask", cleaned)

    # P = pause/resume
    # Q = quit
    key = cv2.waitKey(30) & 0xFF

    if key == ord("p"):
        paused = not paused
    elif key == ord("s"):
        save_parameters()
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
