import cv2
##########################FUNCIONES AUXILIARES###########################
detector = cv2.QRCodeDetector()

def detectar_qr(frame):
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None and data != "":
        # Dibujar recuadro del QR en la imagen
        points = bbox.astype(int).reshape(-1, 2)
        for i in range(len(points)):
            cv2.line(frame,
                     tuple(points[i]),
                     tuple(points[(i + 1) % len(points)]),
                     (0, 255, 0), 2)
        return data 
    return None
SECUENCIA_CORRECTA = ["A", "B", "C", "D"]
buffer = []

def actualizar_buffer(token):
    global buffer
    if token == "Reset Password":
        buffer = []
        print("Buffer reseteado")
        return False
    buffer.append(token)
    print("Secuencia actual:", buffer)

    if buffer == SECUENCIA_CORRECTA:
        print("\n  ¡Desbloqueado!  ")
        buffer = []
        return True
    if len(buffer) > 3:
        buffer = [] # La contraseña esta mal
        print("LA CONTRASEÑA ES ERRONEA")
    return False

def registrar_token(token):
    global ultimo_token

    if token is None:
        return None
    
    elif token != ultimo_token:
        ultimo_token = token
        return token
    
    return None  # QR aún visible → ignorar

########################## PROGRAMA PRINCIPAL ###########################
def desbloquear():
    global ultimo_token, SECUENCIA_CORRECTA, buffer
    SECUENCIA_CORRECTA = ["A", "B", "C", "D"]
    buffer = []

    cap = cv2.VideoCapture(0)
 
    ultimo_token=None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lectura = detectar_qr(frame)
        token = registrar_token(lectura)
        if lectura is not None:
            cv2.putText(frame, f"QR detectado: {lectura}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if token is not None:
            if actualizar_buffer(token):
                break
        cv2.imshow("Sistema de seguridad - QR", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break


    cap.release()
    cv2.destroyAllWindows()




