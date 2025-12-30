import qrcode

qr = qrcode.make("Reset Contraseña")
qr.save("qr_Reset.png")