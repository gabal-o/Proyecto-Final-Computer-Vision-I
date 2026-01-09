import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# Contadores
score1 = 0
score2 = 0
game_over = False

# --- Control de rondas (para no contar continuamente) ---
round_locked = False
unlock_needed = True

def clasificar_gesto(hand_landmarks):
    tips = [8, 12, 16, 20]
    dedos_arriba = []

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            dedos_arriba.append(1)
        else:
            dedos_arriba.append(0)

    if sum(dedos_arriba) == 0:
        return "PIEDRA"
    if sum(dedos_arriba) == 4:
        return "PAPEL"
    if dedos_arriba[0] == 1 and dedos_arriba[1] == 1 and dedos_arriba[2] == 0 and dedos_arriba[3] == 0:
        return "TIJERA"

    return "NADA"

def ganador(j1, j2):
    if j1 == j2:
        return 0
    if (j1 == "PIEDRA" and j2 == "TIJERA") or \
       (j1 == "TIJERA" and j2 == "PAPEL") or \
       (j1 == "PAPEL" and j2 == "PIEDRA"):
        return 1
    return 2

def put_text_fit(img, text, org, max_width, font=cv2.FONT_HERSHEY_SIMPLEX,
                 base_scale=1.0, thickness=2, color=(0,255,0)):
    """
    Escribe texto ajustando el tamaño para que no supere max_width.
    """
    scale = base_scale
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_width:
        scale = max(0.4, base_scale * (max_width / tw))
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def reset_game():
    global score1, score2, game_over, round_locked, unlock_needed
    score1 = 0
    score2 = 0
    game_over = False
    round_locked = False
    unlock_needed = True

cap = cv2.VideoCapture(0)

# --- FPS (suavizado) ---
prev_time = time.time()
fps_smooth = 0.0
alpha = 0.1  # suavizado 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = (1.0 / dt) if dt > 0 else 0.0
    fps_smooth = (1 - alpha) * fps_smooth + alpha * fps

    h, w, _ = frame.shape
    mitad = w // 2

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesto1 = "NADA"
    gesto2 = "NADA"

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            x_hand = hand_landmarks.landmark[0].x * w
            gesto = clasificar_gesto(hand_landmarks)

            # Izquierda → Jugador 1
            if x_hand < mitad:
                gesto1 = gesto
            else:
                gesto2 = gesto

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Línea central
    cv2.line(frame, (mitad, 0), (mitad, h), (255, 255, 255), 2)

    
    txt1 = f"J1: {gesto1} | Victorias: {score1}"
    txt2 = f"J2: {gesto2} | Victorias: {score2}"

    margen = 20
    max_w_left = mitad - 2*margen
    max_w_right = mitad - 2*margen

    put_text_fit(frame, txt1, (margen, 40), max_w_left, base_scale=1.0, thickness=2, color=(0,255,0))
    put_text_fit(frame, txt2, (mitad + margen, 40), max_w_right, base_scale=1.0, thickness=2, color=(0,255,0))

    # FPS + ayuda de controles (arriba)
    fps_text = f"FPS: {fps_smooth:.1f}  |  R: Reset  |  ESC: Salir"
    put_text_fit(frame, fps_text, (20, 80), w - 40, base_scale=0.9, thickness=2, color=(255, 255, 0))

    # Lógica de rondas
    if not game_over:
        ambos_nada = (gesto1 == "NADA" and gesto2 == "NADA")
        ambos_validos = (gesto1 != "NADA" and gesto2 != "NADA")

        if ambos_nada:
            round_locked = False
            unlock_needed = False

        if ambos_validos and (not round_locked) and (not unlock_needed):
            r = ganador(gesto1, gesto2)
            if r == 1:
                score1 += 1
            elif r == 2:
                score2 += 1

            round_locked = True
            unlock_needed = True

        if score1 == 3 or score2 == 3:
            game_over = True

    if game_over:
        texto = "GANADOR: JUGADOR 1" if score1 == 3 else "GANADOR: JUGADOR 2"
        put_text_fit(frame, texto, (50, h//2), w - 100, base_scale=2.0, thickness=3, color=(0,0,255))
        put_text_fit(frame, "Pulsa R para reiniciar", (50, h//2 + 60), w - 100, base_scale=1.0, thickness=2, color=(0,0,255))

    cv2.imshow("Piedra Papel Tijera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if key in (ord('r'), ord('R')):
        reset_game()

cap.release()
cv2.destroyAllWindows()
