from sklearn.cluster import MiniBatchKMeans
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
    imageResult = clt.cluster_centers_.astype("uint8")[labels]

    # voltar para matriz
    imageResult = imageResult.reshape((h, w, 3))

    # converter de lab para rgb
    imageResult = cv2.cvtColor(imageResult, cv2.COLOR_LAB2BGR)

    cv2.imshow("image", np.hstack([imageOriginal, imageResult]))
    cv2.waitKey(0)
    return