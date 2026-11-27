# 無損加速的物理極限盤點

日期：2026-11-25。RTX 5080 16 GB、驅動 621.14、MossInfer 本機 build（`f4f38f5` + reference patch）、ComfyUI v0.45.4。條件：1024²、CFG 雙 row、`high`／INT8、`balanced` + `keep_vram`、20 步 `dpmpp_2m` + `beta`。「無損」指 velocity／像素逐位元相同。

**結論：取樣每步已達「功耗上限下 GEMM 速率」的約 88%。GPU 全程頂在 360 W 板卡上限（此卡可設定的最大值就是 360 W），這是實際的物理牆。** 剩下的無損空間：kernel 融合約 5～10%（需數天、自寫 kernel），以及 ComfyUI 原生文字編碼器約 0.9 s／張（不在本節點範圍）。

## 每張圖的時間組成（實測）

| 階段 | 時間 | 物理下限 | 差距來源 |
| --- | ---: | ---: | --- |
| 文字編碼（ComfyUI 原生，換提示詞時） | 1.13 s | 約 0.2 s | 權重串流 + PyTorch eager；常駐時仍需 0.47 s |
| 取樣節點 | 6.7 s | 約 5.9 s | 見下節 |
| VAE 解碼 + 存檔 | 0.35 s | — | ComfyUI 原生 |
| 合計 | 8.2 s | | |

## 取樣：每步 310～327 ms

- **GPU kernel 佔 97% 以上。** Python 端每步額外 0.7 ms；kernel 之間的間隙每次評估約 2 ms。CUDA Graph 先前已驗證無收益，原因相同。
- 每個 block 7.5 ms（Nsight Systems）：INT8 GEMM 5.31 ms（71%）、8-bit attention 1.37 ms（18%）、norm／RoPE／量化 0.95 ms（13%）。
- **功耗：** 取樣期間 351～361 W，`SW Power Cap` 持續啟動，時脈 2.73～2.78 GHz（最高 3.09 GHz）。`nvidia-smi -q` 顯示 Max Power Limit = Default = 360 W，無法再調高。
- **GEMM：** 模型內約 355 TOPS。單獨量測各 tile 為 338～390 TOPS（2.8 GHz 下理論峰值的 70～81%）。上游自動選擇的 tile 在 sm_120 上已是最佳（差距在量測雜訊內）。cuBLAS INT8（`torch._int_mm`）同形狀為 315～350 TOPS，沒有更快。只跑暫存器的 mma 微基準為 455～487 TOPS。
- **Roofline：** 每步總運算量約 96.7 TOP（GEMM 78.8 + attention 18）。以功耗上限下的 GEMM 速率 355 TOPS 計算，下限為 272 ms，實測 310 ms，達 88%。

INT8 GEMM 是整數累加，換 tile／kernel／split-K 都不會改變位元，只要 epilogue 的浮點運算順序不變。所以 GEMM 是唯一能「隨意換」的無損區塊。但它已經和 cuBLAS 一樣快，又被功耗上限壓住時脈。

## 已實驗並否決

| 方案 | 結果 |
| --- | --- |
| tile 重新調校（sm_120） | 現行自動選擇已是最佳，不需修改 |
| kept engine 重用時跳過 `unload_all_models` | 像素相同，文字編碼 1.13 → 0.87 s；但 VRAM 衝到 15.7 GB，其中一張取樣從 6.6 s 變 8.2 s（WDDM 換頁）。不穩定，已還原 |
| 文字編碼器與 DiT 同時常駐 | 9.3 + 7.1 GB 加上桌面約 2.8 GB，超過 16 GB，屬於 VRAM 容量限制 |

## 剩餘選項（依收益排序，皆未實作）

1. **attention 前處理融合進 QKV GEMM epilogue**：`norm_rope_stats`、`quant_int8`、`v_fp8_transpose` 每 block 合計 0.51 ms（約佔每步 7%）。epilogue 的 N=256 tile 剛好涵蓋兩個 head，RMS norm 與 RoPE 可以在 tile 內完成，預估省 4～5%。風險：RMS 平方和必須沿用原 kernel 的歸約順序才能逐位元相同。工作量約數天。
2. **sm_120 專用的 warp-specialized 持久 INT8 GEMM**（epilogue 與 mainloop 重疊）：GEMM 效率 70～81% → 約 85%。在功耗上限下，時間收益預估只有 3～6%。工作量約一週。
3. **硬體端（使用者自行操作）**：降壓（V/F curve）或加強散熱，可在同樣 360 W 下跑更高時脈。本次觀察到 GPU 變熱後每步從 0.309 s 漂移到 0.327 s（約 6%），未做受控量測。降壓不穩定時可能產生靜默計算錯誤，必須重新驗證位元。
4. **ComfyUI 原生文字編碼器**：常駐時仍需 0.47 s（9B 模型、約 30 tokens，頻寬下限約 10 ms），推測是 eager 模式的 Qwen3.5 GDN 層開銷。屬於 ComfyUI 核心，不在本節點範圍。

不在本文範圍：減少步數、換採樣器、改精度（都會改變圖片，見 [採樣數學研究](MATH_ACCELERATION_RESEARCH.md)）。

## 重現

腳本與原始資料位於 `test_results/physical_limit/`（已被 `.gitignore` 排除）：`bench.py`（直接驅動 moss-server，逐步計時，計算每 row velocity 的 SHA-256）、`peak.py`／`sustain.py`（cuBLAS 峰值與功耗監測）、`mma_peak.cu`（tensor core 上限）、`e2e.py`／`te_only.py`（ComfyUI 逐節點計時）、`prof_bal.nsys-rep`。tile 排名用上游 `mossinfer_tests.exe --gtest_filter=GpuQuantKernels.Throughput`，需設定 `MI_RUN_PERF=1 MI_QUANT_PERF_WINDOW_MS=1500`。
