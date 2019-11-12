import cv2
import socket

def udp_server(ip="127.0.0.1"):

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(type(udp))
    udp.connect((ip, 12345))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),50]
    #キャプチャに使うカメラの選択
    capture = cv2.VideoCapture(0)

    while True:
       
        _,frame = capture.read()
        # UDPで送る為に画像をstringに変換
        _,jpgstring = cv2.imencode(".jpg", frame ,encode_param)
        jpgstring = jpgstring.tostring()
        udp.send(jpgstring)
        
    return

udp_server()
