"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

project_root = os.path.dirname(os.path.realpath(__file__))
category_embedding = [[0.9479751586914062, 0.4561353325843811, 0.16707628965377808], [0.1,-0.1,0.1], [0.5455077290534973, -0.6193588972091675, -2.629554510116577], [-0.1,0.1,-0.1]]


systemRoot = "/data/xuzhenbo/"
kittiRoot = os.path.join(systemRoot, "kitti/")
MOTS20Root = os.path.join(systemRoot, "MOT17/")
MOTRoot = os.path.join(systemRoot, "MOT17/")
rootDir = os.path.join(systemRoot, 'CCP')
apolloRoot = "/data/xuzhenbo/apollo_kitti/"
apolloOriRoot = "/data/xuzhenbo/apollo/"
apolloCrop = os.path.join('/home/xuzhenbo/','crop_apollo')
if os.path.isdir(os.path.join(systemRoot, "anaconda3/envs/dense3D")):
    pythonPath = os.path.join(systemRoot, "anaconda3/envs/dense3D/bin/python")
elif os.path.isdir("/home/xuzhenbo/anaconda3/envs/pt17"):
    pythonPath = "/home/xuzhenbo/anaconda3/envs/pt17/bin/python"
else:
    pythonPath = '/home/xuzhenbo/.conda/envs/dense3D/bin/python'
