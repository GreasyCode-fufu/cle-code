# Source and result integrity report

- Teacher SHA256 before/after: `39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04` / `39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04` — unchanged: **True**.
- Ultralytics core aggregate SHA256 before/after: `7a7f53d45cebecee365d94c320db459d903384ab9128bcfe52a735f22a1e5ad1` / `7a7f53d45cebecee365d94c320db459d903384ab9128bcfe52a735f22a1e5ad1` — unchanged: **True**.
- All completed best.pt and last.pt files free of Teacher/KD/discriminator keys or modules: **True**.
- Student inference complexity fixed at 2,603,129 fused / 2,607,913 unfused parameters and 7.044096 GFLOPs: **True**.
- Selected checkpoints load and infer without `ultralytics_ext`: **True**.

Protected pre-existing results:

- `runs/detect/hongkong_SR1_P4Only_RelKD`: unchanged = **True** (`da860cd9db5128f4da1b4fcc734a3bc4e125498bb9ef82be856223ee9c8d96d0`).
- `runs/detect/hongkong_SR1_P4Only_PairedNoKD`: unchanged = **True** (`0aa52171da0b181dbf7102cce333e468330e76eb2f7527b6cd82bc8f3528994b`).
- `runs/detect/hongkong_SR1_P4Only_MTRD`: unchanged = **True** (`5aa577658651c7232fe715e50f839b67454a160fd17dbdb01ae1ee2908af5c6a`).
- `runs/analysis/kd_formal_result`: unchanged = **True** (`89643061edb109ff593d03fe12704f995b91667b96bea5b4c9a045cda0e8824c`).
- `runs/analysis/kd_paired_nokd_result`: unchanged = **True** (`fbb6ce8bb89e5c952997cd6c9871cb21e940851d95ae780173c442960a984667`).
- `runs/analysis/mtrd_formal_result`: unchanged = **True** (`eff749daefb89aac3d4e4183c0e5a65e7ab434a0e28e31e288d046436d23e53f`).
