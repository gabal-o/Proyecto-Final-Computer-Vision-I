import qrcode

qr = qrcode.make("Reset Password")
qr.save("qr_Reset.png")