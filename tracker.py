import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# Contadores
score1 = 0
score2 = 0
game_over = False

def clasificar_gesto(hand_landmarks):
    tips = [8, 12, 16, 20]  # Dedos índices
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # Mostrar textos
    cv2.putText(frame, f"J1: {gesto1}  | Victorias: {score1}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"J2: {gesto2}  | Victorias: {score2}", (mitad + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.line(frame, (mitad, 0), (mitad, h), (255, 255, 255), 2)

    if not game_over:
        if gesto1 != "NADA" and gesto2 != "NADA":
            r = ganador(gesto1, gesto2)
            if r == 1:
                score1 += 1
            elif r == 2:
                score2 += 1

            # Pausa pequeña para evitar repetir lecturas
            cv2.waitKey(500)

        if score1 == 3 or score2 == 3:
            game_over = True

    if game_over:
        texto = "GANADOR: JUGADOR 1" if score1 == 3 else "GANADOR: JUGADOR 2"
        cv2.putText(frame, texto, (50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    cv2.imshow("Piedra Papel Tijera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
