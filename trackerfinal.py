import cv2
import numpy as np
from collections import deque, Counter
import time


def majority_vote(history, min_votes=8):
    """Return the most common label if it appears at least min_votes times; else 'NADA'."""
    if not history:
        return "NADA"
    c = Counter(history)
    label, votes = c.most_common(1)[0]
    return label if votes >= min_votes else "NADA"

def winner(j1, j2):
    if j1 == j2:
        return 0
    if (j1 == "PIEDRA" and j2 == "TIJERA") or \
       (j1 == "TIJERA" and j2 == "PAPEL") or \
       (j1 == "PAPEL" and j2 == "PIEDRA"):
        return 1
    return 2

def preprocess_diff_adaptive_bg(roi_bgr, bg_float, kernel,
                               diff_thresh=25,
                               blur_ksize=(5, 5)):
    """
    Adaptive background subtraction pipeline:
    absdiff(bg, roi) -> gray -> blur -> FIXED threshold -> morphology
    Returns: mask (uint8), gray_diff (uint8)
    """
    bg_u8 = cv2.convertScaleAbs(bg_float)
    diff = cv2.absdiff(bg_u8, roi_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)

    # Umbral
    _, mask = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)

    # Morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask, gray

def best_contour(mask, min_area=7000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue

        solidity = area / hull_area
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 1:
            continue
        extent = area / (w * h)

        score = area * (0.6 * solidity + 0.4 * extent)
        if score > best_score:
            best_score = score
            best = c

    return best

def count_defects_and_hull(contour):
    area = cv2.contourArea(contour)
    if area < 7000:
        return 0, None

    eps = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)

    hull_points = cv2.convexHull(approx, returnPoints=True)
    hull_idx = cv2.convexHull(approx, returnPoints=False)

    if hull_idx is None or len(hull_idx) < 3:
        return 0, hull_points

    defects = cv2.convexityDefects(approx, hull_idx)
    if defects is None:
        return 0, hull_points

    n = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = approx[s][0]
        end = approx[e][0]
        far = approx[f][0]

        depth = d / 256.0

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b < 1e-6 or c < 1e-6:
            continue

        cosang = (b*b + c*c - a*a) / (2*b*c)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))

        if angle <= 80 and depth > 18:
            n += 1

    return n, hull_points

def classify(defects, area):
    if area < 7000:
        return "NADA"
    if defects == 0:
        return "PIEDRA"
    if defects in (1, 2):
        return "TIJERA"
    if defects >= 3:
        return "PAPEL"
    return "NADA"

# -----------------------------
# Main
# -----------------------------
def tracker():
    cv2.setUseOptimized(True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Two ROIs
    ROI_W, ROI_H = 260, 260
    ROI1_X, ROI1_Y = 60, 80
    ROI2_X, ROI2_Y = 360, 80

    # Adaptive background models (float32)
    bg1 = None
    bg2 = None

    # Adaptation controls
    alpha_bg = 0.01       # velocidad de adaptación del fondo 
    diff_thresh = 15        # umbral fijo de diferencia 
    update_white_ratio = 0.03  # si la máscara tiene <3% blanco, asumimos "sin mano" y actualizamos fondo

    # Game state
    score1, score2 = 0, 0
    game_over = False
    round_locked = False

    last_round_text = ""
    show_round_until = 0.0

    hist_len = 15
    min_votes = 8
    h1 = deque(maxlen=hist_len)
    h2 = deque(maxlen=hist_len)

    VALID = {"PIEDRA", "PAPEL", "TIJERA"}

    prev_time = time.time()
    fps_ema = 0.0
    alpha = 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * inst_fps

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        roi1 = frame[ROI1_Y:ROI1_Y + ROI_H, ROI1_X:ROI1_X + ROI_W]
        roi2 = frame[ROI2_Y:ROI2_Y + ROI_H, ROI2_X:ROI2_Y + ROI_H,] if False else frame[ROI2_Y:ROI2_Y + ROI_H, ROI2_X:ROI2_X + ROI_W]

        cv2.rectangle(frame, (ROI1_X, ROI1_Y), (ROI1_X + ROI_W, ROI1_Y + ROI_H), (0, 255, 0), 2)
        cv2.rectangle(frame, (ROI2_X, ROI2_Y), (ROI2_X + ROI_W, ROI2_Y + ROI_H), (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Inicialización del background con B
        if bg1 is None or bg2 is None:
            cv2.putText(frame, "Press B with empty ROIs to init adaptive background", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "ESC exit | B background | R reset score", (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("RPS (OpenCV only)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key in (ord('b'), ord('B')):
                bg1 = roi1.astype(np.float32)
                bg2 = roi2.astype(np.float32)
                round_locked = False
                h1.clear()
                h2.clear()
            elif key in (ord('r'), ord('R')):
                score1, score2 = 0, 0
                game_over = False
                round_locked = False
                h1.clear()
                h2.clear()
            continue

        # 1) Background adaptativo + 2) Umbral fijo
        m1, _ = preprocess_diff_adaptive_bg(roi1, bg1, kernel, diff_thresh=diff_thresh)
        m2, _ = preprocess_diff_adaptive_bg(roi2, bg2, kernel, diff_thresh=diff_thresh)

        # Actualiza fondo SOLO si parece "sin mano" (pocos píxeles blancos)
        white1 = cv2.countNonZero(m1) / float(ROI_W * ROI_H)
        white2 = cv2.countNonZero(m2) / float(ROI_W * ROI_H)

        if white1 < update_white_ratio:
            cv2.accumulateWeighted(roi1, bg1, alpha_bg)
        if white2 < update_white_ratio:
            cv2.accumulateWeighted(roi2, bg2, alpha_bg)

        # Find hand contours
        c1 = best_contour(m1, min_area=7000)
        c2 = best_contour(m2, min_area=7000)

        g1 = "NADA"
        g2 = "NADA"

        if c1 is not None:
            area1 = cv2.contourArea(c1)
            d1, hull1 = count_defects_and_hull(c1)
            g1 = classify(d1, area1)
            cv2.drawContours(roi1, [c1], -1, (0, 255, 0), 2)
            if hull1 is not None:
                cv2.polylines(roi1, [hull1], True, (255, 255, 255), 2)

        if c2 is not None:
            area2 = cv2.contourArea(c2)
            d2, hull2 = count_defects_and_hull(c2)
            g2 = classify(d2, area2)
            cv2.drawContours(roi2, [c2], -1, (0, 255, 0), 2)
            if hull2 is not None:
                cv2.polylines(roi2, [hull2], True, (255, 255, 255), 2)

        # Stabilize gestures
        h1.append(g1)
        h2.append(g2)
        g1s = majority_vote(h1, min_votes=min_votes)
        g2s = majority_vote(h2, min_votes=min_votes)

        # Game logic
        now = time.time()
        if not game_over:
            if (not round_locked) and (g1s in VALID) and (g2s in VALID):
                r = winner(g1s, g2s)
                if r == 1:
                    score1 += 1
                    last_round_text = "Round: Player 1 wins"
                elif r == 2:
                    score2 += 1
                    last_round_text = "Round: Player 2 wins"
                else:
                    last_round_text = "Round: Draw"

                show_round_until = now + 1.0
                round_locked = True

            if round_locked and (g1s == "NADA") and (g2s == "NADA"):
                round_locked = False

            if score1 >= 3 or score2 >= 3:
                game_over = True

        # HUD
        cv2.putText(frame, f"P1: {g1s} | {score1}", (ROI1_X, ROI1_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"P2: {g2s} | {score2}", (ROI2_X, ROI2_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if now < show_round_until:
            cv2.putText(frame, last_round_text, (20, H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        if game_over:
            win_text = "WINNER: PLAYER 1" if score1 >= 3 else "WINNER: PLAYER 2"
            cv2.putText(frame, win_text, (20, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            cv2.putText(frame, "Press R to reset | Press B to re-init background",
                        (20, H // 2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "ESC exit | B background | R reset score", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("RPS (OpenCV only)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('b'), ord('B')):
            # re-init adaptive background
            bg1 = roi1.astype(np.float32)
            bg2 = roi2.astype(np.float32)
            round_locked = False
            h1.clear()
            h2.clear()
        elif key in (ord('r'), ord('R')):
            score1, score2 = 0, 0
            game_over = False
            round_locked = False
            h1.clear()
            h2.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker()
