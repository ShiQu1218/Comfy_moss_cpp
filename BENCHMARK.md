# C++ 節點與原生 ComfyUI 速度比較

2026-11-15，本機 RTX 5080 16 GB，Windows，驅動 621.14，ComfyUI 0.44.2、Python 3.12.10、PyTorch 2.10.0+cu128，啟用 `--fast fp16_accumulation`。C++ 引擎為 MossInfer 2.2.1 CUDA 13。

共同設定：**50 步、1024×1024、ER-SDE、simple、CFG 4、shift 5、batch 1**。正向提示詞為 `A red apple on a wooden table, natural window light, realistic photography, detailed texture`，負向為空白。重新載入測試種子 42，快取測試種子 43。

兩邊使用同一份 `moss_image_sft_int8_convrot.safetensors`、`qwen3.5_9b_int8_convrot.safetensors`、`flux2-vae_small.safetensors`。負向分支全程跳過第 22 層；原生流程透過 `SkipLayerGuidanceDiTSimple` 對齊 C++ 預設。沒有 LoRA、prompt enhancer 或額外步驟快取。

## 結果

| 情境 | C++ 節點總耗時 | 原生 ComfyUI 總耗時 | C++ 速度倍率 |
| --- | ---: | ---: | ---: |
| 卸載模型後，第 1 輪 | 23.428 秒 | 103.630 秒 | 4.42× |
| 卸載模型後，第 2 輪 | 22.647 秒 | 86.059 秒 | 3.80× |
| 兩輪平均 | **23.037 秒** | **94.845 秒** | **4.12×** |
| 保留快取、更換種子再生成 | **22.251 秒** | **83.823 秒** | **3.77×** |

重新載入兩輪平均減少約 75.7% 總耗時；連續生成減少約 73.5%。時間涵蓋 workflow 執行至 SaveImage 完成，不包含測試前卸載模型的等待時間。

C++ 引擎回報兩輪採樣耗時分別為 18.619、18.781 秒，平均 18.700 秒。原生 KSampler 節點分別耗時 93.390、83.797 秒；該節點時間包含模型初始化／準備，因此不能視為與 C++ 純採樣時間完全相同的計時範圍。總 workflow 時間是主要比較依據。

## 測試方式與產物

- 兩條流程依序在同一個獨立 ComfyUI GPU 程序執行，停用其他自訂節點，只載入本節點。原本的 ComfyUI 沒有同時執行工作；測試前已釋放其模型快取。
- 前兩輪各次都要求卸載模型與清除 ComfyUI 執行快取；沒有清除 Windows 檔案快取，因此不是磁碟完全冷啟動測試。首輪原生流程另有初始化成本。
- 快取測試緊接原生第 2 輪，保留模型及文字 conditioning，將種子換成 43，確認 KSampler、C++ 節點與 SaveImage 都有實際執行。C++ 節點的架構每次仍會啟動子程序並重新載入模型。
- 透過 websocket 的 `execution_success` 事件計時，驗證六次執行都成功，六張 PNG 都是有效的 1024×1024 圖片。已目視檢查首輪兩張圖片，皆符合紅蘋果提示詞。
- 原始事件、history、日誌及整理後的 `summary.json` 位於 `test_results/benchmark/`。重現指令見 README 的 `scripts/benchmark.py`。
- [C++ workflow](workflows/benchmark_cpp.json)、[原生 workflow](workflows/benchmark_native.json) 可直接拖進 ComfyUI；同目錄也保留 API JSON。
- 首輪圖片位於 ComfyUI `output/moss_benchmark_cpp_00001_.png` 與 `output/moss_benchmark_native_00001_.png`。

## 解讀限制

C++ 使用 `high`／`min_memory`，其中 attention 為 8-bit；原生使用目前 PyTorch 的預設 attention 與記憶體管理。兩邊的隨機數與數值實作也可能不同，圖片並不逐像素相同。本結果是這台機器與目前軟體環境下的實際工作流程速度，不能推論為相同精度的核心運算加速倍率。

未升級 PyTorch、調校原生後端或測試其他模型、長提示詞與 GPU。重新載入僅各兩次，保留快取僅各一次，不代表完整統計分布。

節點已新增 sampler／scheduler 輸入，舊 workflow 預設仍為 euler／moss；4 項單元測試通過。原本的 ComfyUI 需重新啟動才能載入新增選項。專用測試伺服器已關閉。
