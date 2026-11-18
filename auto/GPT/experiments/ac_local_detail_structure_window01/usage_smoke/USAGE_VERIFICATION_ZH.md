# 首实际权重使用核验

本轮事前冻结nativeCUDA卷积后端、原AMP FP16、真实物理B1。固定首val2386、mom_detailcontrast seed45最早valbest2真实权重，单图对原B32保存概率最大差7.845461368560791e-5，小于事前1e-4，类别一致。未更换图/权重/阈值，未填充B32，未访问新test图。

实际权重SHA d53335233b78f9c638ef195776e968c5d639de6bad0e4f3af33110774f98b655，服务器再次独立读字节重算通过，本地未下载权重字节。

work/verify_local_detail_usage_archive.py本地逐文件SHA、冻结cfg/source、全部427val原CSV、60epoch/最早best2/每epoch2077、图SHA、权重引用及实际概率差均passed。这仅是计算使用验证，不能当三seed新增贡献、所有图误差保证、独立泛化或临床验证。16文件原包及receipt保持不变。
