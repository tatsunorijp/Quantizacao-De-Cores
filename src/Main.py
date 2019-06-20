import argparse
import Functions

# argumentos do programa
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required=True, type=int,
                help="# of clusters")
args = vars(ap.parse_args())


# Functions.kmeans(args)
Functions.kmeansCV(args)