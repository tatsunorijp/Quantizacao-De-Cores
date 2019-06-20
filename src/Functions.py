from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
import cv2

def kmeans(args):
    # carrega a imagem e pega suas dimensoes
    imageOriginal = cv2.imread(args["image"])
    (h, w) = imageOriginal.shape[:2]

    # converter do sistema RGB para o sistema l*a*b
    imageResult = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2LAB)

    # transformar em uma matriz de 1 dimensão, para aplicar o k-means
    imageResult = imageResult.reshape((imageOriginal.shape[0] * imageOriginal.shape[1], 3))

    # aplicar o k-means para a quantidade de cluster e
    # criar a imagem quantizada baseado nas predicoes, cores da paleta
    clt = MiniBatchKMeans(n_clusters=args["clusters"])
    labels = clt.fit_predict(imageResult)
    print(labels)
    imageResult = clt.cluster_centers_.astype("uint8")[labels]

    # voltar para matriz
    imageResult = imageResult.reshape((h, w, 3))

    # converter de lab para rgb
    imageResult = cv2.cvtColor(imageResult, cv2.COLOR_LAB2BGR)

    cv2.imshow("image", np.hstack([imageOriginal, imageResult]))
    cv2.waitKey(0)
    return

def kmeansCV(args):
    img = cv2.imread(args["image"])
    imageResult = img.reshape((-1, 3))

    imageResult = np.float32(imageResult)

    K = args["clusters"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


    ret, label, center = cv2.kmeans(imageResult, K, None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS)#cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    imageResult = res.reshape((img.shape))

    cv2.imshow('res2', np.hstack([img, imageResult]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()