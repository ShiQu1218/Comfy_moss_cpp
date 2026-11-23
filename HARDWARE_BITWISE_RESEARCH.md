# 嚴格逐位元等價的硬體加速評估

## 實作與實驗結果

獨立採樣器已新增 `cuda_graphs` 進階選項。共用記憶體改用 NumPy 緩衝區檢視，輸出仍複製到獨立緩衝區，以免下一次去噪覆寫。LoKr helper 重用 CUDA 工作區與 ConvRot 表，每批從 128 列增為 512 列，沿用原本的運算順序與同步 CPU 備援。

RTX 5080、SFT INT8、`high`／`q8_fp8`、1024×1024、ER-SDE／simple、50 步、seed 42、CFG 4。各模式跑三次，輪替順序。耗時涵蓋採樣節點與引擎啟動，扣除雜湊比對時間。VRAM 為引擎回報峰值，VAE 另行驗證。

| 設定 | 中位數（範圍） | VRAM 峰值 | 實際權重配置 |
| --- | ---: | ---: | --- |
| `min_memory` | 22.17 秒（22.10–25.18） | 1.38 GiB | 串流，每次評估約 4.44 GiB |
| `balanced` | 19.72 秒（19.62–19.76） | 6.75 GiB | 常駐 |
| `balanced` + `cuda_graphs` | 19.70 秒（19.64–19.71） | 6.75 GiB | 常駐 |

九次執行中，50 次評估的 sigma、rows、輸入 latent 與每列 velocity 的 SHA-256 均相符。最終 latent、ComfyUI VAE 輸出的浮點資料與 RGB8 也相符。`balanced` 的中位數耗時減少 11.1%。Graph 組與單用 `balanced` 的時間範圍重疊。

共享記憶體變更完成後，以相同條件再跑三種路徑各一次。每次評估與最終輸出的雜湊都與修改前的 `min_memory` 基準一致。這輪用於驗證輸出，未分離 CPU 複製耗時。

另以 512×512、12 步、seed 43、SLG 切換與單一 LoKr 跑三種路徑。每次評估與最終輸出均相符，引擎確認套用一個 adapter。Nsight Systems 在同一個引擎程序記錄到 84 次成功的 Graph instantiate 與 183 次成功的 Graph launch，包含 99 次重播。此案例的 `min_memory` 也選擇常駐，實際配置應以當次引擎計畫為準。

LoKr 完整模型合併以同一份約 595 MB 的 adapter 與同一份 5.4 GB INT8 DiT，交錯跑新版、舊版 GPU helper 各三次。計時含磁碟讀寫，排除 SHA-256 計算。中位數由 **11.52 秒降至 10.65 秒**（少 7.6%），六份輸出模型的 SHA-256 完全相同。新版另通過五種 ConvRot 分組的 GPU／強制 CPU 整檔逐位元比對，包含正負 LoKr 與完整 diff。新版已置於本機 `runtime/`。既有快取會依 helper 雜湊在下次使用時重建。

38 項 Python 單元測試通過，包含共享記憶體對負零、次正規數、NaN payload 的位元保留與輸出緩衝區隔離。

可用 ComfyUI 的 Python 重現，並在本地 `test_results/bitwise/` 查閱 `main.json`、`after_copy.json`、`merge.json`、`slg_lokr.json` 與 `slg_trace.nsys-rep`。

```powershell
python scripts/benchmark_bitwise.py D:/ComfyUI/ComfyUI_windows_portable/ComfyUI test_results/bitwise/main.json
```

## 原始研究

日期：2026-11-16。研究目前工作區與固定版本上游原始碼；未修改程式、執行檔或 workflow。

## 結論與比較基準

優先順序：**驗證權重常駐 → 啟用既有 CUDA Graph → 改善 LoKr 合併的配置／傳輸 → 減少 Python 共享記憶體複製**。目前沒有證據支持直接購買新硬體，或承諾新的加速倍率。

本報告的 baseline 是目前專案在同一台機器、相同軟體／權重／輸入下，`forecast_denoise=False` 的既有 C++ 路徑；不是原生 PyTorch。`BENCHMARK.md` 已說明兩者圖片不同，既有 4.12× 不能當成 bitwise-equivalent 加速證據。若要求相對原生 PyTorch 完全相同，目前專案尚未滿足此前提。

嚴格等價須固定權重與 adapter 內容、順序、強度，conditioning、輸入 latent、初始 noise、完整 sigma 序列、採樣器與 RNG 狀態，以及 tier／attention／dtype。驗收比對每次評估的 velocity rows 與最終 latent 的原始位元組，若宣稱整條出圖流程等價，另比對 VAE 結果與 RGB。不可只用 allclose、PSNR、目視或 PNG 檔案雜湊；PNG metadata／壓縮差異也可能改變檔案內容。

固定 GPU、driver、引擎 DLL、cuBLASLt、PyTorch、編譯選項與 kernel route；不能把「新版更準」當成等價。NVIDIA 對 cuBLAS 的可重現性有 toolkit、GPU 架構／SM 數量及並行執行條件限制，跨版本／換卡不自動保證。[官方說明](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

## 本機與已確認證據

- RTX 5080，16,303 MiB VRAM，Windows WDDM，driver 621.14。
- Ryzen 7 7700，8 核心／16 執行緒；系統可見 RAM 約 31.2 GiB。
- 專案所在 D: 為 Kingston KC3000 2 TB NVMe；其他位置的模型檔案是否在同一顆磁碟未逐一確認。
- 現有模組化測試曾記錄 1024²／50 步總時間 23.537 秒，採樣節點含準備 21.133 秒（`MODULAR_TEST_REPORT.md`）；不是本次重新量測。
- 完整 LoKr 模型 CPU 50.955 秒、GPU 17.986 秒，輸出檔相同，是既有單次紀錄（`test_results/gpu_merge/comparison.json`），不能據此推估未實作最佳化的效益。

本次實際執行 `tests/check_gpu_merge.py`，使用 ComfyUI 的 Python。兩個分解 LoKr（正、負強度）加完整 diff，五種 ConvRot 分組皆通過整份 safetensors 位元組相等，並確認 GPU 路徑有執行。受測 build-gpu helper 與 runtime helper 的 SHA-256 相同。

| ConvRot group | 強制 CPU（秒） | GPU（秒） | 位元組相等 |
| --- | ---: | ---: | --- |
| 4 | 0.067 | 0.180 | 是 |
| 16 | 0.056 | 0.163 | 是 |
| 64 | 0.077 | 0.169 | 是 |
| 256 | 0.137 | 0.161 | 是 |
| 1024 | 0.359 | 0.184 | 是 |

這是每次啟動 helper 的單模組測試，含啟動／I/O，沒有多輪統計或完全隔離背景工作。可證明這些案例正確，也顯示小工作量不一定適合 GPU；不能直接據此設計每層 CPU/GPU 分派門檻。腳本目前以同一份新版 helper 的強制 CPU 分支作對照，不是另一份歷史 CPU 執行檔。

## 可行方向，依優先順序

### 1. 用既有 balanced 模式驗證權重常駐

`sampler.py:100` 預設 `min_memory`，`:131` 原樣交給 server。上游 `src/planner/planner.h:10` 說明該模式優先降低 VRAM，可能每步搬移 DiT blocks；balanced／max_speed 優先保留可放入的權重。

先比較 min_memory 與 balanced，再視結果比較 max_speed；保留相同 high／INT8／attention。只改資料放置，不主動改數學運算，這是最小且最值得先驗證的方向。上游已有 `tests/unit/test_dit_stream.cpp:234` 的 resident／streamed 逐位元測試，但本機目前工作流程仍須驗收。

記錄實際 residency、resident_blocks、stream_bytes、權重等待時間與 VRAM 峰值。16 GB 不等於保證所有尺寸、長提示詞、SLG、LoRA 都能常駐；只看 5.4 GB 權重檔會漏掉 activation、工作區及兩個程序的 CUDA context。也不能假定目前每一次 min_memory 都必定串流；須讀實際 plan。

### 2. 接上已存在、但目前節點沒要求的 CUDA Graph

**只把 memory_mode 改成 balanced 不會自動啟用 Graph。** `sampler.py:131` 的 load 沒有 `block_path`；上游 `include/mossinfer/mossinfer.h:236` 與 `docs/server-protocol.md:165` 指定預設是 `fused`，不是 `fused_graphs`。

最小實驗是在 load 使用既有 `block_path: "fused_graphs"`，不必重寫 kernel 或增加依賴。`src/models/moss_dit/moss_dit.cpp:1420` 要求全權重常駐、允許捕捉、形狀重複等條件才真正 replay；串流時會退回 eager fused。應確認 graph_replays 大於零，並比對 eager／graph 的各步位元組，不能只看設定名稱。

上游已有圖快取、位址／shape／LoRA generation 的 key、fused 路徑自我檢查；應沿用。SLG rows 切換可能增加 capture；短工作量可能無法攤平首次捕捉成本。這一項與權重常駐有關聯，量測時要分開比較，不能把收益相加。

### 3. LoKr helper 重用 GPU 工作區，再考慮傳輸重疊

`native/lokr_merge.cu:81` 每個 module 都建立多個 Device 配置，逐 128 列執行同步 cudaMemcpy；`native/lokr_cache.cpp:57` 再逐 module 呼叫。首次建立快取才受益，快取命中後不會因此加快 DiT 採樣。

先量測並嘗試重用 GPU 工作區、Hadamard 表，減少 cudaMalloc/free 與固定上傳。再評估有限大小的 pinned staging 與兩組緩衝區，讓不同列塊的傳輸／計算重疊。不同列塊可獨立排程；同一列的 adapter 順序、rotate 的 j 累加順序、量化與取整都必須保留。NVIDIA 說明傳輸／計算重疊需要適當 pinned memory、streams 與硬體支援；不能只把函式名稱換成 Async。[官方指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)

保留現有「整個 module 成功後才提交 CPU 輸出」機制，否則 GPU 失敗後 CPU fallback 可能重複套用 patch。事件須涵蓋錯誤旗標與最後 D2H 完成；勿鎖定整個模型占用大量 RAM。

`rotate` 是每個輸出依 j 逐項 FP32 累加，可研究資料重用／shared-memory tiling，但必須保留逐項運算順序。不能直接改成 butterfly FWHT、平行加總或 Tensor Core GEMM；這些會改變 rounding。`--fmad=false` 也不能消除重新排列加總的差異。[浮點運算與 FMA 說明](https://docs.nvidia.com/cuda/floating-point/index.html)

### 4. 減少 Python 端複製；零拷貝 C API 列為後續

`component.py:127` 每次評估將 x 從 PyTorch GPU 複製到 CPU，再經 tobytes 寫入 mapping；結果先切出 bytes、NumPy copy，再送回 PyTorch GPU。上游 `src/pipeline/session.cpp:833` 已將共享 staging 註冊為 pinned，`:999`／`:1079` 則負責引擎端 H2D／D2H。因此「新增共享記憶體」或「替 server staging 加 pinned」不是新機會。

可先以 mapping 的 buffer view 減少多餘 CPU bytes 配置，保留必要的 ownership copy／同步，避免下一次 denoise 覆寫仍在使用的輸出。1024² 的單個 FP32 latent 是 2 MiB；兩個輸出 rows 時，每次評估兩側程序合計約 12 MiB 的 host/device 傳輸。50 次約 600 MiB，不含其他資料；實際次數依採樣器而異。故不能未量測就聲稱 PCIe 頻寬是主瓶頸，同步延遲可能更重要。

上游 `docs/c-api.md:606` 已支援 MI_MEMORY_DEVICE 與 stream event handoff。同程序直接呼叫 C API 可消除跨 CPU 的往返，但需要處理 CUDA context／runtime 共存、tensor lifetime、取消和錯誤恢復；目前 PyTorch cu128 與引擎 cu13 的整合需另外驗證。不能將子程序的 device pointer 直接交給另一程序。這不是第一個應做的改動；也不要為省拷貝把既有 PyTorch sampler／CFG 改寫到 C++，那會擴大數值驗證範圍。

### 5. 模型常駐程序、CPU 因子計算與磁碟 I/O

- `sampler.py:153` 每次建立 server，`:191` 結束 session；`component.py:170` 直接回收程序。重用 server 可攤平載入、context 與圖捕捉成本，但會改變 VRAM 釋放行為，須與後續原生 VAE／TE 協調。模型、adapter、shape、guidance 及取消後狀態都要正確失效；列為第二階段，不先新增常駐管理架構。
- 上游 `models/lora/lora_merge.cpp:36` 的因子展開採 double 順序累加。若 profile 顯示它是瓶頸，可平行化獨立的輸出列／欄並保留每個元素的 inner 順序；不直接換一般 GEMM、FP32 或重排 reduction。也須先確認編譯器是否已做欄方向 SIMD，避免重做現有最佳化。
- helper 先整檔複製再逐 module 讀／改／寫（`native/lokr_cache.cpp:54`）。較快 SSD／更多 RAM 的頁面快取主要幫助冷載入與首次合併；D: 已是 NVMe，應先量測讀寫／換頁，不能保證換硬體有效。64 GB RAM 是減少壓力的候選，不是已證明的升級需求。
- 更大 VRAM 可減少串流，但換 GPU 可能改變 kernel 選擇與運算結果。嚴格維持此台 5080 baseline 時，不能把換卡列為已符合 bitwise 的方案。

## 不可當成嚴格等價的捷徑

- forecast_denoise、TeaCache／FirstBlockCache 等跨步近似、少算步數或變更 SLG。
- high 改 turbo／NVFP4／INT4、更換 attention 精度、TF32、fast-math、改 FMA 或累加精度。
- 任意融合 CFG／sampler、改 batch／row packing、split-K、多 GPU reduction。固定 seed 或開 deterministic mode 都不代表與原 baseline 相同。
- DoRA 的 CPU／GPU 路徑互換。`lokr_cache.py:153` 的首次快取使用 GPU 原生計算，而 `lora.py:50` 預設 CPU；既有 `DORA_CACHE_REPORT.md` 已記錄非零差異。快取開／關與 fallback 不能先視為等價。此次 helper 測試比較的是「相同輸入 diff」的合併，沒有證明 DoRA diff 的產生過程相同。

## 最小驗收實驗

1. 封存 baseline 的輸入原始位元組、模型／DLL 雜湊與設定；先在完全相同條件重跑兩次，確認 baseline 本身穩定。
2. 分開比較 A=min_memory+fused、B=balanced+fused、C=balanced+fused_graphs；不順便改 cache、tier、attention、採樣器或任何 RNG 行為。
3. 保存每次評估的 sigma、rows、velocity 與最終 latent，逐位元組比對；任何一位不同即不通過。包含 seed 42/43、512²/1024²、長短 conditioning、CFG 單／雙分支、SLG 切換、一般 LoRA、LoKr 及 DoRA，並針對取消／失敗後重跑確認狀態。
4. 同時保留實際 plan、kernel route、graph_replays、VRAM 峰值、準備／推論／傳輸時間。冷啟動與熱執行分開，各條件交錯量測至少數輪，報中位數與範圍，不以單次較快判定。
5. helper 最佳化沿用現有 byte 比對，補 group=0、尾端列塊、不同 k、零值／極小值／量化半整數邊界及非有限值拒絕；保留 GPU 失敗後 CPU fallback 測試。若動核心算術，除案例測試還須檢查逐元素運算順序／編譯結果，不能聲稱有限測試證明所有輸入。

本次完成程式碼／既有量測分析、硬體盤點及現有 helper 的五組逐位元測試；未執行新一輪完整出圖、三種模式 A/B/C、Nsight profile 或跨裝置驗證。上述新方向均是待驗證候選，尚無新的效能提升或端到端等價保證。
