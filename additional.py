#!/usr/bin/python
import cv2
import numpy as np
import imutils
import random
import dlib
import socket
from contextlib import closing

def land2coords(landmarks, dtype="int"):
    # タプルのリストを初期化 (x,y)座標
    coords = np.zeros((68, 2), dtype=dtype)
    # ランドマークをループし、それらを変換? (a,b)座標の2組
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # a,b座標のリストを返す
    return coords

# srcTriとdstTriを使って計算したアフィン変換をsrcに適用し、サイズの画像を出力
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # 一対の三角形が与えられれば、アフィン変換を求める
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # 見つかったアフィン変換をsrcイメージに適用
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# 点が矩形内にあるかどうかをチェック
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#　ドロネー三角形を計算
def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    # サブディビジョンにポイントを挿入
    for p in points:
        subdiv.insert(p) 
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    pt = []    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1 
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        pt = []        

    return delaunayTri
        

def warpTriangle(img1, img2, t1, t2) :

    # 各三角形の境界矩形を見つける
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # それぞれの矩形の左上隅のオフセット点
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # 三角形を塗りつぶしてマスクを得る
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
    
    # 小さな長方形のパッチにwarpImageを適用する
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #print(img1Rect) #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask
    # 長方形パッチの三角領域を出力画像にコピーする
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 

# main部分
if __name__ == '__main__':

    # ローディングdlibのHogベースの顔検出器
    face_detector = dlib.get_frontal_face_detector()
    # dlibのdatファイルを読み込む
    landmark_predictor = dlib.shape_predictor("test.dat")

    buff = 1024
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip = "127.0.0.1"
    sock.bind((ip, 12345))
    while True:
        try:

            jpgstring,addr = sock.recvfrom(buff*1280)   #送られてくるデータが大きいので一度に受け取るデータ量を大きく設定
            narray = np.fromstring(jpgstring, dtype = "uint8")  #string型からnumpyを用いuint8に戻す
            decimg = cv2.imdecode(narray,1) #uint8のデータを画像データに戻す
            imgf = decimg
            imgb = cv2.imread("任意の画像を追加してください.jpg")
            img1Warped = np.copy(imgb)
            
            '''一つ目の動画処理(背景)'''
            frame_back = cv2.imread("任意の画像を追加してください.jpg")
            # cv2.resizeも使えるがimutilsのほうが調整楽
            frame_back = imutils.resize(frame_back, width=1180)  # 顔のサイズ？
            # 計算上効率的のためグレースケール画像で操作を実行する
            frame_gray = cv2.cvtColor(frame_back,cv2.COLOR_BGR2GRAY)
            # 顔を検出(この次のfor文の中で顔を切り抜いてやる？) face_boundaries:<class 'dlib.dlib.rectangles'>
            face_boundaries = face_detector(frame_gray,0)
            landmarks_b = ""
            landmarkslist = ""
            point_b = []
            # 顔の座標を取り出してlandmarkslistに格納(enumerateで値を分割している)
            for (enum,face) in enumerate(face_boundaries):
                landmarks_b = landmark_predictor(frame_gray, face)
                landmarks_b = land2coords(landmarks_b)
                landmarkslist_b = landmarks_b.tolist()
            # サブディビジョンにポイントを挿入する
            for (num,p) in enumerate(landmarkslist_b) :
                point_b.append(tuple(p))
            
            '''二つ目の通信によってやってきた動画処理(顔画像)'''
            frame_front = imutils.resize(decimg, width=620) # 顔の部分の縮尺(小さいほうが大きい)
            frame2_gray = cv2.cvtColor(frame_front,cv2.COLOR_BGR2GRAY)
            frame2_boundaries = face_detector(frame2_gray,0)
            # 顔の判定をとれなかった時のための処理
            if len(frame2_boundaries) == 0:
                print("NoFaces")
                continue
        
            landmarks2 = ""
            landmarkslist2 = ""
            point_f = []
            # 顔の座標を取り出してlandmarkslistに格納(enumerateで値を分割している)
            for (enum,face) in enumerate(frame2_boundaries):
                landmarks2 = landmark_predictor(frame2_gray, face)
                landmarks2 = land2coords(landmarks2)
                landmarkslist2 = landmarks2.tolist()
            # サブディビジョンにポイントを挿入する
            for (num,p) in enumerate(landmarkslist2) :
                point_f.append(tuple(p))
            # triangle系統はx,y座標ごちゃ混ぜに<class 'numpy.ndarray'>
            '''ここらへんからアフィン変換'''
            # 三角形の座標に変換
            points1 = point_f
            points2 = point_b

            # 凸包を見つける
            hull1 = []
            hull2 = []
            # 背景画像のconvex hull上の点に対応する輪郭上の点のインデックス
            hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
                
            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            # 背景画像の凸包点のドロネー三角形を求める
            sizeImg2 = imgb.shape    
            rect = (0, 0, sizeImg2[1], sizeImg2[0])
            dt = calculateDelaunayTriangles(rect, hull2)
            if len(dt) == 0:
                quit()
            # ドロネー三角形にアファイン変換を適用
            for i in range(0, len(dt)):
                t1 = []
                t2 = []
                # 三角形に対応するimg1、img2の点を取得
                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])
                #print(t1)print(t2) [(177, 200), (135, 196), (127, 189)]
                warpTriangle(imgf, img1Warped, t1, t2)
            
            # マスクを計算
            hull8U = []
            for i in range(0, len(hull2)):
                hull8U.append((hull2[i][0], hull2[i][1]))
            
            mask = np.zeros(imgb.shape, dtype = imgb.dtype)  
            
            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
            
            r = cv2.boundingRect(np.float32([hull2]))    
            
            center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
            
            # シームレスにクローン
            output = cv2.seamlessClone(np.uint8(img1Warped), imgb, mask, center, cv2.NORMAL_CLONE)
            
            cv2.imshow("Face Swapped", output)
            cv2.waitKey(30)
        
        except :
            print("out")
            continue