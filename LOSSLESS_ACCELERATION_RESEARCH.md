本次研究結論：**不改變出圖的加速，最大來源是「每張圖重開引擎」的固定成本。讓 `moss-server` 在兩張圖之間保持常駐後，`run_begin` 從 2.5～3.3 秒降到 0.02～0.46 秒，velocity 逐位元相同。** 再搭配既有的 `balanced` 模式（每步約快 16%，同樣逐位元相同），20 步的採樣理論上可從約 10.4 秒降到約 6.4 秒。

日期：2026-11-25。上游 MossInfer commit `f4f38f5`，RTX 5080。已實作為採樣器進階選項 `keep_engine`，並以真實工作流程驗收（見文末）。

## 目前每張圖的時間組成

`dpmpp_2m` + `beta` + 20 步、1024²、CFG 4，取自 [NFE 掃描](MATH_ACCELERATION_RESEARCH.md) 的伺服器日誌：

| 階段 | 秒數 | 說明 |
| --- | ---: | --- |
| 啟動 `moss-server` | 0.13 | 每張圖重開一次程序 |
| `run_begin` | 2.25 | 載入 DiT、配置記憶體 |
| 第一次評估 | 0.44 | 比一般步驟多約 0.08 秒暖機 |
| 其餘 19 步 | 約 7.6 | 每步約 0.40 秒 |
| 合計採樣 | 10.4 | |

步數降到 10～20 步後，固定成本的佔比變大，10 步時約佔 35%。

## 依據

上游 `include/mossinfer/mossinfer.h` 說明：引擎會保留常駐的 DiT，相同條件的下一次執行不重新載入（`mi_memory_stats.dit_loads 0`），並「給出相同的圖片位元」；LoRA 不變時也會沿用。`moss-server` 另提供 `release {"keep_host_copy"}`，可以釋放 VRAM，同時保留引擎（必要時保留主機 RAM 的權重副本）。

本專案目前在每次採樣後關閉程序（`component.py::ComponentSession.close`），所以每張圖都要重新付出載入成本。

## 實測

腳本：`test_results/persist/check_persistent.py`、`check_release.py`。使用合成 conditioning 與固定 x，1024²，5 個 sigma，每步計算正負兩個 row，把所有 velocity 位元串起來做 SHA-256 比對。

| 模式 | 新程序 `run_begin` | 同程序第 2 次 | `release` 後再執行 | 5 步耗時（首次／重用） |
| --- | ---: | ---: | ---: | ---: |
| `min_memory` | 2.48～2.54 s | 0.03 s | 0.37 s（release 0.01 s） | 2.35／1.85 s |
| `balanced` | 3.27～3.30 s | 0.02 s | 0.46 s（release 1.05 s） | 1.60／1.55 s |

- **逐位元一致：** 同一 seed 重複執行、新程序對照重用程序、`release` 前對照 `release` 後，velocity 雜湊全部相同。`min_memory` 和 `balanced` 的雜湊也彼此相同，與 [硬體研究](HARDWARE_BITWISE_RESEARCH.md) 的結論一致。
- `min_memory` 下 `release` 沒有保留主機副本：引擎回報「5.0 GiB 會讓可用 RAM 低於 8 GiB」，但重新載入仍只需 0.37 秒（推測受益於作業系統檔案快取，未證實）。
- `balanced` 常駐時約佔 7.2 GB VRAM；`release` 需要約 1 秒把 5.4 GB 權重複製到 RAM。

## 可選方案

| 方案 | 20 步採樣預估 | VRAM 佔用（兩張圖之間） | 風險 |
| --- | ---: | --- | --- |
| 現況（每張重開） | 約 10.4 s | 無 | 無 |
| A：常駐，每張圖後 `release` | 約 8.5 s | 幾乎無 | 低 |
| B：常駐，不釋放，`min_memory` | 約 8.0 s | 約 1.5 GB | ComfyUI 不知道這塊 VRAM 被佔用 |
| C：常駐，不釋放，`balanced` | 約 6.4 s | 約 7.2 GB | 同上，其他模型較容易 OOM |

以上預估把實測的 `run_begin` 差額、暖機與每步時間套到 20 步，尚未以真實工作流程量測。

**建議：** 做一個預設關閉的進階選項 `keep_engine`。開啟時，以 `load` 參數（模型、DiT 檔、tier、mode、LoRA 快取檔等）作為快取鍵，重用同一個 session；每張圖結束後呼叫 `release`（方案 A），讓 ComfyUI 載入其他模型時不受影響。另一個選項可以讓使用者選擇不釋放（方案 B／C）。

實作須處理：

1. 每次執行後呼叫 `conditioning_free` 釋放這次匯入的 conditioning，否則常駐程序會累積記憶體。
2. 發生錯誤或取消時，維持現行行為，關閉並丟棄 session；下一張圖再重開。
3. 快取鍵改變時（換模型、LoKr 快取檔、tier、mode），先關閉舊程序。
4. ComfyUI 結束時，子程序在 stdin 關閉時會自行結束（上游 server 的 end-of-input 行為）。
5. 驗收：以真實工作流程連續產圖，比對開關兩種情況下的 latent／PNG 是否逐位元相同，並量測每張圖的時間。

## 限制

- 只用合成 conditioning、5 個 sigma 驗證；尚未以真實提示詞、LoRA 切換、尺寸切換或 reference latent 連續執行。
- 換尺寸或 LoRA 時，引擎可能部分重新載入，節省會較少。
- 其他不改變出圖的既有項目：完整 SLG 共用前段約 1.2×，只適用完整 SLG，需要改引擎；LoKr 合併的 butterfly 只影響首次建立快取。見 [數學等價研究](MATH_EQUIVALENCE_RESEARCH.md)。

## 實作與驗收（2026-11-25）

`sampler.py` 新增進階選項 `keep_engine`（`off`／`release_vram`／`keep_vram`，預設 `off`），以 `[executable, load]` 作為快取鍵，最多保留一個程序。每次成功採樣後呼叫 `ComponentSession.finish`，釋放本次的 conditioning；`release_vram` 另外呼叫 `release`。失敗、取消或快取鍵改變時，關閉舊程序。快取鍵包含 DiT 檔的大小與修改時間。使用 LoKr 磁碟快取的圖一律不保留程序，因為 `cache/lokr/merged.safetensors` 是固定路徑，保留會沿用舊權重，Windows 上還會鎖住重建。`import_conditioning` 與 reference latent 在引擎複製資料後立即 `shm_free`，避免常駐程序累積到 scratch 上限。單元測試新增 9 項，`tests/test_modular.py` 共 25 項全數通過。清理失敗時保留已完成的圖，只丟棄引擎；改走原生採樣時先關閉常駐引擎。

真實工作流程：`dpmpp_2m` + `beta` + 20 步、1024²、seed 42，依序產生文字、版面、照片、文字共 4 張，五種組合連續執行。腳本見 `test_results/persist/run_keep_engine.py`。

| keep_engine／memory_mode | 第 1 張採樣 | 第 2～4 張採樣 | 每張圖後 VRAM（含其他程序） |
| --- | ---: | ---: | ---: |
| off／min_memory | 10.48 s | 10.36～10.41 s | 3.3 GB |
| release_vram／min_memory | 9.76 s | 7.83～7.84 s | 3.5 GB |
| keep_vram／min_memory | 7.45 s* | 7.63～7.68 s | 5.0 GB |
| off／balanced | 9.86 s | 9.58～9.81 s | 3.3 GB |
| keep_vram／balanced | 9.54 s | 6.41～6.42 s | 10.4 GB |

\* 沿用了前一組 `release_vram` 留下的同鍵程序。

- **像素逐位元相同：** 21 張圖（含暖機）中，同一提示詞的解碼 RGB 雜湊在五種組合之間、以及同組重複出圖時完全一致。PNG 檔案本身的雜湊不同，是因為 SaveImage 會把含檔名前綴的 workflow 寫進中繼資料，所以改比對像素。
- 第 2 張起，`release_vram` 快 25%，`keep_vram` + `balanced` 快 38%。
- **LoRA 切換：** 普通 LoRA（`MOSS_彩色線稿`，246 個 side branch）依「有、無、有、無」連續出圖，`off`、`keep_vram`、`release_vram` 三種模式的像素雜湊兩兩相同；有 LoRA 與無 LoRA 的結果不同，代表沒有沿用舊狀態。腳本見 `run_lora_switch.py`。
- 未測：LoKr 快取（設計上不保留程序）、尺寸切換、reference latent 連續執行，以及長時間連續產圖的記憶體穩定性。
