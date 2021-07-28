## PointTrackV2 && CCP


This codebase implements **PointTrackV2 (TPAMI 2021)** and **CCP(ICCV 2021)**, a highly effective framework for multi-object tracking and segmentation (MOTS) described in: 

```
@ARTICLE{9449985,
  author={Xu, Zhenbo and Yang, Wei and Zhang, Wei and Tan, Xiao and Huang, Huan and Huang, Liusheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Segment as Points for Efficient and Effective Online Multi-Object Tracking and Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3087898}}
@inproceedings{xu2021continuous,
  title={Continuous Copy-Paste for One-stage Multi-object Tracking and Segmentation},
  author={Xu, Zhenbo and Meng, Ajin and Yang, Wei and Huang, Liusheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6222--6231},
  year={2019}
}
```

**PointTrackV2 presents a new learning strategy for pixel-wise feature learning on the 2D image plane, which has proven to be effective for instance association.**

Our network architecture adopts [SpatialEmbedding](https://github.com/davyneven/SpatialEmbeddings) as the segmentation sub-network. 
The current ranking of PointTrack is available in [KITTI leader-board](http://www.cvlibs.net/datasets/kitti/eval_mots.php). Until now (07/03/2020), PointTrack++ still ranks first for both cars and pedestrians.
The detailed task description of MOTS is avaliable in [MOTS challenge](https://www.vision.rwth-aachen.de/page/mots).  


## Getting started

This codebase showcases the proposed framework named PointTrack for MOTS using the KITTI MOTS dataset. 

### Prerequisites
Dependencies, please refer to 'pt17.yml' 

Note that the scripts for evaluation is included in this repo. After images and instances (annotations) are downloaded, put them under **kittiRoot** and change the path in **repoRoot**/config.py accordingly. 
The structure under **kittiRoot** should looks like:

```
kittiRoot
│   images -> training/image_02/ 
│   instances
│   │    0000
│   │    0001
│   │    ...
│   training
│   │   image_02
│   │   │    0000
│   │   │    0001
│   │   │    ...  
│   testing
│   │   image_02
│   │   │    0000
│   │   │    0001
│   │   │    ... 
```

## Contact
If you find problems in the code, please open an issue.

For general questions, please contact the corresponding author Wei Yang (qubit@ustc.edu.cn).


## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).




