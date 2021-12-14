### Dataset

Download the [images](#) and [gts](#) of all the datasets.

#### Training

DUTS_class + COCO-seg (9K)

#### Testing

CoCA + CoSOD3K + CoSal2015.



### Run

Run locally: `./noCls_gco.sh`

Submit multiple experiments to DGX server: `./sub_by_id.sh`



### Evaluation

The performance is saved in `./evaluation` folder.

`cd evaluation` and `python sort_results.py` to show best results in a comprehensive way.



### Current Performance

> Metrics: E-measure (max) +, S-measure +, Fmax +, MAE -

|            Methods \ Datasets             |        CoCA (ECCV2020)        |      CoSOD3K (CVPR2020)       |     CoSal2015 (CVPR2015)      |
| :---------------------------------------: | :---------------------------: | :---------------------------: | :---------------------------: |
|                 **Ours**                  | 0.828 / 0.754 / 0.660 / 0.077 | 0.903 / 0.850 / 0.831 / 0.059 | 0.925 / 0.884 / 0.890 / 0.053 |
| GCoNet (no COCO-seg) (CVPR2021, baseline) | 0.760 / 0.673 / 0.544 / 0.105 | 0.860 / 0.802 / 0.777 / 0.071 | 0.887 / 0.845 / 0.847 / 0.068 |
|              CADC (ICCV2021)              |   - / 0.681 / 0.548 / 0.132   |   - / 0.801 / 0.759 / 0.096   |   - / 0.866 / 0.862 / 0.064   |
|              ICNet(NIPS2020)              | 0.698 / 0.651 / 0.506 / 0.148 | 0.832 / 0.780 / 0.743 / 0.097 | 0.900 / 0.856 / 0.855 / 0.058 |
|      CoSformer (arXiv, Transformer)       | 0.770 / 0.724 / 0.603 / 0.103 | 0.879 / 0.835 / 0.807 / 0.066 | 0.929 / 0.894 / 0.891 / 0.047 |

