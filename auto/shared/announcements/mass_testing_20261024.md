# 库模块量产阶段(用户指令)
1. 并发上限: 单Agent<=3, 队列深度>=6, GPU不停
2. module_library 425个模块: 用 GatedWrap 模式(zcode工作区 models/optimize/lib_variants.py
   可参考)快速包装成零初始化门控变体批量试; 已证伪家族(通道注意力/裸乘性注意力/
   频域重标定/方向条带)优先跳过
3. zcode 首批: haar小波/方向上下文/线性注意力(Batch L 已启动); 各家自行挑模块认领
