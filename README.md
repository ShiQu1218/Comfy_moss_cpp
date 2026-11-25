# Comfy_moss_cpp

將 [mossimage.cpp / MossInfer](https://github.com/mossimage-cpp/mossimage.cpp) 接到 ComfyUI 的獨立採樣節點。使用 ComfyUI V3 API，沿用既有原生節點與標準 `MODEL`、`CONDITIONING`、`LATENT` 連線。

## 速度

![MOSS C++ 採樣速度比較](docs/speed.png)

每張圖的採樣時間，從採樣節點開始計到 VAE Decode 開始；keep_engine 的數據取第 2 張起。

- **橘色：** 改用 `dpmpp_2m` + `beta`，可把步數降到 20 步；10～12 步時文字與版面仍正確，照片細節略降。出圖會和預設的 ER-SDE 不同。見 [採樣數學研究](MATH_ACCELERATION_RESEARCH.md)。
- **藍色：** 開啟進階選項 `keep_engine`，連續生圖時省去每張重新載入 DiT 的時間，像素與 20 步逐位元相同。設定方式見下方「參考圖」一節之後的 `keep_engine` 說明，數據見 [無損加速研究](LOSSLESS_ACCELERATION_RESEARCH.md)。

重新產生圖表：`python docs/speed_chart.py`。

## 接入現有 workflow

新增 **KSampler (MOSS Image C++)**，替換原本的 KSampler，接回同樣的四組輸入與 latent 輸出：

| 新節點插槽 | 接既有節點 |
| --- | --- |
| `model` | Load Diffusion Model 的 MOSS-Image MODEL |
| `positive` / `negative` | CLIP Text Encode 的正／負 CONDITIONING |
| `latent_image` | 官方空白潛在影像（EmptyLatentImage）、EmptySD3LatentImage、EmptyFlux2LatentImage 或原生 VAE Encode 的 LATENT |
| LATENT 輸出 | 原本的 VAE Decode，再接 SaveImage |

不用重新選 text encoder、VAE、尺寸或輸入 prompt。Diffusion model 的檔案來自輸入 MODEL；文字與 VAE 仍由原生 ComfyUI 處理。種子、步數、CFG、採樣器、scheduler 與 denoise 屬於採樣器設定；和原生 KSampler 一樣，可將 widget 轉成輸入，接既有的數值節點。shift 沿用 MODEL 的 model_sampling。

Load Diffusion Model 的 `weight_dtype` 請選 `default`。低 denoise 的離散排程可能重複選到同一個 sigma，此時只執行有變化的步驟。

進階選項 `forecast_denoise` 預設關閉。開啟後，實驗性的去噪預測會依 sigma 步距，用最近兩次完整推論的速度張量外推少數步驟，減少 MossInfer 呼叫。前段、末段、SLG 分支切換時及步距不適合外推時會執行完整推論。速度取決於採樣器與排程，預測可能改變文字或圖像細節。主控台會顯示實際預測的次數。

官方空白 latent 會沿用原生 KSampler 的 `fix_empty_latent_channels`，依 MODEL 將全零 latent 的通道與縮放比例轉成 MOSS 格式，保留原生尺寸轉換規則，不會修改輸入資料。非空白的其他模型 latent 不會被強制當成 MOSS latent。範例：[empty_latent.json](workflows/empty_latent.json)。目前仍限 batch 1；MOSS 的圖片尺寸須配合 16 倍數。

[workflows/modular.json](workflows/modular.json) 是完整可匯入範例，[modular_api.json](workflows/modular_api.json) 是 API 版本。預設為 1024×1024、50 步、ER-SDE、simple、CFG 4。舊的 `MossImageCpp` 整合節點保留，顯示名稱改為 Legacy All-in-One，舊 workflow 不需要改寫。

### LoRA

直接沿用原生 **Load LoRA** 或 **Load LoRA (Model Only)**，將它的 MODEL 接到 C++ 採樣器即可；支援多個 LoRA 串接、正／負 `strength_model` 與 alpha。使用 Load LoRA 時，CLIP 輸出照常接正／負 CLIP Text Encode，`strength_clip` 由原生文字編碼流程處理，不會在 C++ 再套一次。採樣器不需要另外選 LoRA。

[workflows/lora.json](workflows/lora.json) 與 [lora_api.json](workflows/lora_api.json) 提供原生 Load LoRA 的完整範例。範例使用本機的 `moss_黑白線稿.safetensors`，請換成自己的 MOSS-Image LoRA；不會自動下載。

支援的轉送方式：

| 格式 | 處理方式 |
| --- | --- |
| 一般 LoRA（含原生已識別的 kohya、PEFT、diffusers 等命名） | 原低秩張量、alpha、正／負強度交給 C++ |
| 線性 LoHa／LoKr | 保留原因子與原生 alpha 規則，交給 C++ |
| DoRA、LoCon、Tucker、同形狀 reshape，以及其他原生 WeightAdapter | 用原生 ComfyUI 計算該參數的完整 patch 堆疊，再轉成差值交給 C++ |
| 完整 diff、bias diff | 交給 C++ 套用 |

完整差值模式會讀取未套用 patch 的原始參數，依原順序處理堆疊與非線性 DoRA；不會再重複套用同一參數的低秩 patch。逐參數計算與寫檔，完成、失敗或取消後清除專屬暫存檔。它會比低秩模式更吃 RAM、暫存磁碟與載入時間；INT8 合併也可能增加重新量化誤差，不能保證逐像素等同原生 PyTorch。

基礎模型與原 LoRA 檔案不會被修改。若原生轉換失敗、產生非有限值、改變模型參數形狀，或 C++ 回報未識別的 adapter，會停止輸出。仍需使用與 MOSS-Image 架構相容的 LoRA；其他模型的卷積／改變通道數等 adapter 不會因此變成相容。

驗證見 [LORA_TEST_REPORT.md](LORA_TEST_REPORT.md)；數值回歸測試可用 ComfyUI 的 Python 執行 `tests/test_lora.py`。

若採樣前等待較久，可查看主控台的 `MossInfer preparation`：分別記錄卸載、LoRA 匯出、server 初始化、run_begin 與首次評估耗時。進度條會先顯示 0%，載入完才開始計入採樣步數。LoHa／LoKr 在此引擎可能需要載入時合併，INT8 還要重新量化；單一作用範圍大的 adapter 就可能花數十秒，不能只按啟用檔案數預估成本。LoKr 現可使用下述磁碟快取，避免重複合併。Power Loader 比對結果見 [POWER_LORA_PROFILE.md](POWER_LORA_PROFILE.md)。

### 跳層引導（SLG）

可直接保留原生 **SkipLayerGuidanceDiTSimple** 與 **SkipLayerGuidanceDiT**，把 MODEL 輸出接到 C++ 採樣器：

- Simple：只在原本的 sigma 區間內，替負向分支跳過指定層。
- 完整 SLG：依原 scale 增加跳層正向分支，保留 start／end percent 與 rescaling_scale。支援多個完整 SLG 串接，也能與 Simple、LoRA 同時使用。
- MOSS 只有 `single_block`，因此 `single_layers` 才會跳層；`double_layers` 與原生 MOSS 一樣沒有作用。每層有效索引為 0～41。
- 依引擎上限，一個 component session 最多 8 個 prediction rows（含正／負分支與額外跳層分支）；超過會明確報錯。

[workflows/slg.json](workflows/slg.json)／[slg_api.json](workflows/slg_api.json) 展示原生 Load LoRA → Simple → 完整 SLG → C++ 採樣器，其他文字與 VAE 節點不需更換。驗證見 [EXTENDED_TEST_REPORT.md](EXTENDED_TEST_REPORT.md)。目前辨識的是本機 ComfyUI 的這兩種原生 SLG callback，其他插件的任意 MODEL callback 仍會拒絕。

此介面使用 MossInfer component session，透過共享記憶體匯入原生 conditioning 與 latent，由原生 ComfyUI 採樣演算法執行迴圈、C++ 執行 DiT，不是重新拼回文字生圖命令。因此先前整合節點的速度數據不能直接套用。

目前支援原生 Load Diffusion Model 的 MOSS-Image、上述 adapter 與原生 SLG、單張 batch、單組正／負 conditioning、無 mask 的 latent，以及下述參考圖；不支援 ControlNet、區域或時間排程 conditioning、其他 MODEL callback。遇到這些設定會明確報錯，不會靜默忽略。其他模型請使用原生採樣器。

### 參考圖（ReferenceLatent）

原生 **ReferenceLatent** 的參考 latent 可交給 C++ 引擎：參考圖的 KV 在 run_begin 時計算一次，之後每一步重用。條件如下，不符合時會在主控台警告 `using the native sampler` 並改用原生 KSampler，不會報錯：

- 正、負 conditioning 帶相同的參考 latent，且 `reference_latents_method` 為 `index_timestep_zero_text_start`（Moss-Image-Edit 的位置排列）。
- 最多 4 張，合計不超過 16384 個 latent token；不能與 MOSS SLG 同時使用。
- 需要套用 [patches/mossinfer-2.2.1-reference-latents.patch](patches/mossinfer-2.2.1-reference-latents.patch) 的 `moss-server`（`hello` 回報 `run_reference_latents`）。官方 2.2.1 套件沒有這項功能，會自動退回原生採樣。

本機 RTX 5080、1024×1024、20 步、CFG 4、一張 768×1024 參考圖：原生 62.5 秒，C++ 13.3 秒。與原生輸出的相對差距和無參考圖時的 INT8 誤差同級，可用 `scripts/check_reference_latents.py` 以合成輸入重測。

#### 編譯參考圖版引擎

```powershell
git clone https://github.com/mossimage-cpp/mossimage.cpp
cd mossimage.cpp
git checkout f4f38f5e90d9990795d125b357574a68aadbe924
git apply <本 repo>/patches/mossinfer-2.2.1-reference-latents.patch
cmake --preset windows-msvc-cuda13 -DCMAKE_CUDA_ARCHITECTURES=120-real   # 依顯示卡調整
cmake --build build/windows-msvc-cuda13 --target moss-server mossinfer
```

將產生的 `moss-server.exe` 與 `mossinfer.dll` 覆蓋官方套件 `runtime/<套件名稱>/bin/` 內的同名檔（建議先備份），其餘 DLL 沿用官方套件。

進階選項包含 bundle 設定目錄、tier、memory_mode 與 cache_lokr。上游 component API 即使只跑 DiT 仍會檢查文字編碼器與 VAE 的標頭，因此本機仍需有相容的 `qwen3.5_9b*` 文字編碼器與 `flux2*` VAE 元件檔；會從標準模型搜尋路徑選取第一個符合名稱的檔案供標頭檢查，不會以它們取代輸入的 conditioning 或執行解碼。這些模型的安裝方式如下。

VRAM 足夠時，可將獨立採樣器的 `memory_mode` 設為 `balanced`，讓引擎優先將 DiT 權重留在 GPU。另可開啟進階選項 `cuda_graphs`，在全部 DiT 權重常駐且推論形狀重複時啟用 CUDA Graph。若引擎選擇串流權重，則沿用一般 fused 路徑。`cuda_graphs` 預設關閉。逐位元比對與本機量測方式見 [硬體加速報告](HARDWARE_BITWISE_RESEARCH.md)。

進階選項 `keep_engine` 可在連續生圖時保留 C++ 引擎程序，省去每張圖重新載入 DiT 的時間，輸出像素與關閉時逐位元相同。設定如下：

| 設定 | 兩張圖之間 | 本機 20 步採樣（第 2 張起） |
| --- | --- | ---: |
| `off`（預設） | 每張結束即關閉程序 | 10.4 s |
| `release_vram` | 程序保留，DiT 釋放 VRAM，下張重新載入約 0.4 s；仍佔約 0.2 GB | 7.8 s |
| `keep_vram` | DiT 留在 VRAM：`min_memory` 約 1.6 GB、`balanced` 約 7.1 GB | 7.7 s／6.4 s（balanced） |

`release_vram` 在 RAM 足夠時會把 DiT 副本留在主機 RAM（本機 `balanced` 約 5.4 GB；`min_memory` 因 RAM 不足而未保留）。`keep_vram` 佔用的 VRAM 不受 ComfyUI 管理，同一個 ComfyUI 再載入其他大型模型時可能 OOM。更換 DiT（含同路徑替換檔案）、tier、memory_mode 或 cuda_graphs，或採樣失敗、取消時，會關閉舊程序並重新啟動。使用 LoKr 磁碟快取的那張圖不保留程序：快取固定覆寫同一個檔案，保留會用到舊權重並鎖住重建。數據與驗證見 [無損加速研究](LOSSLESS_ACCELERATION_RESEARCH.md)。

### LoKr 載入快取

獨立採樣器預設開啟進階選項 `cache_lokr`。第一次把 LoKr 合併至一份獨立的 DiT 檔；後續相同模型、LoKr 組合及強度直接載入這份檔案，跳過每張圖重做的合併與 INT8 重新量化。更換提示詞、seed、尺寸、步數或 SLG 不需重建。沿用原生或 rgthree Power LoRA Loader 的接法即可，原始模型不會被修改。

快取只保留最近一組，位於本節點的 `cache/lokr/`，占用一份 DiT 的磁碟空間；建立新組合時暫時再需要一份。修改 LoKr 內容／強度、來源模型路徑／大小／修改時間或 helper 版本會重建。不在 RAM／VRAM 常駐合併模型；重啟 ComfyUI 仍可重用磁碟快取。Ampere 及更新的 GPU 會照常即時套用普通 LoRA，包含區塊 side branch，以及與 LoKr 作用於不同權重的文字嵌入、時間調節與輸出層。舊 GPU 上混用一般 LoRA 與 LoKr、其他非線性 adapter 混用、特殊 patch、turbo 或不相容權重會沿用原本載入方式。快取失敗時會警告並退回原流程。

這份磁碟快取不會在採樣結束時刪除；關閉 `cache_lokr` 也只停止使用快取。需要釋放磁碟空間時，先停止所有正在使用此節點的工作，再刪除 `cache/lokr/`。下次啟用時會重新合併。C++ 引擎程序及其 RAM／VRAM 預設於每次採樣結束後回收（`keep_engine` 開啟時除外）。

需要本機 helper `runtime/moss-lokr-cache.exe`（Linux 不含 `.exe`），此工作區已編譯安裝。它直接編譯固定版本上游的 CPU LoKr 合併與重新量化函式，原樣複製模型標頭與未修改的張量。重建方式見 [native/README.md](native/README.md)；未安裝 helper 時會提示並使用原流程。

## 安裝

1. 將本目錄放在 `ComfyUI/custom_nodes/Comfy_moss_cpp`，重新啟動 ComfyUI。
2. 從上游下載符合作業系統、顯示卡與驅動的 [MossInfer release](https://github.com/mossimage-cpp/mossimage.cpp/releases/tag/v2.2.1)，核對 `SHA256SUMS` 後，將完整套件解壓到本目錄的 `runtime/`。預期位置為 `runtime/<套件名稱>/bin/moss-infer.exe`（Linux 為 `moss-infer`）。只放一個版本；也可在啟動 ComfyUI 前設定環境變數 `MOSSIMAGE_CPP_EXECUTABLE` 指向可信任的執行檔。不可只複製 exe，還需要套件的 DLL / 函式庫。
3. 準備官方 MOSS-Image Base 或 SFT 的設定與 tokenizer。以下只下載小檔案，不下載權重；將路徑換成自己的 ComfyUI：

```powershell
hf download OpenMOSS-Team/MOSS-Image-SFT --revision 2c7954437269461767f533b006823e8f85973a60 --include '*/config.json' --include 'tokenizer/*' --include 'model_index.json' --include 'scheduler/*' --local-dir 'D:/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/mossimage/MOSS-Image-SFT'
```

4. 將相容的元件權重放入標準模型目錄：

| 元件 | 目錄 | 範例 workflow 使用的檔名 |
| --- | --- | --- |
| DiT | `models/diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` |
| 文字編碼器 | `models/text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors` |
| VAE | `models/vae/` | `flux2-vae_small.safetensors` |

設定目錄與元件須符合相同架構；上游引擎會拒絕不支援的模型。現有 `unet` 等 ComfyUI 設定的模型搜尋路徑也會沿用。不需要額外 Python 套件，使用 ComfyUI 已有的 torch、numpy、Pillow。

## 舊整合節點與測試

把 [workflows/txt2img.json](workflows/txt2img.json) 拖進 ComfyUI，確認模型選項後執行。流程為 `MOSS Image (mossimage.cpp) → SaveImage`，固定種子 42、512×512、12 步，適合快速驗證；正式產圖可提高到 1024×1024、50 步。

API 版本為 [workflows/txt2img_api.json](workflows/txt2img_api.json)，可用 ComfyUI 的 Python 執行：

```powershell
python scripts/test_workflow.py --url http://127.0.0.1:8188
$env:COMFYUI_PATH = 'D:/ComfyUI/ComfyUI_windows_portable/ComfyUI'
python tests/test_nodes.py
```

節點提供 `sampler` 與 `scheduler`，舊 workflow 預設沿用 `euler`／`moss`。執行後會在 ComfyUI 日誌輸出引擎各階段耗時。

速度比較使用 [C++ workflow](workflows/benchmark_cpp.json) 與 [原生 workflow](workflows/benchmark_native.json)：50 步、1024×1024、ER-SDE、simple、CFG 4、shift 5，使用相同的 INT8 權重、提示詞與種子。兩邊的負向分支都跳過第 22 層，沒有 LoRA 或額外加速快取。測試前先確認其他 GPU 工作已結束。

```powershell
python scripts/benchmark.py --url http://127.0.0.1:8189 --runs 2
```

此命令會在指定的 ComfyUI 上依序測試，先各跑兩輪（每輪清除 ComfyUI 執行快取並卸載模型），再保留原生模型快取改用種子 43 重跑，最後以種子 43 測試 C++ 節點。作業系統的檔案快取不清除。每次記錄 websocket 執行事件、整體時間與輸出至 `test_results/benchmark/`；若採樣或輸出節點被快取跳過，測試會失敗。C++ 節點每次都會重新載入模型，因此不具備原生流程的常駐模型優勢。

測試程式實際提交 workflow，檢查執行狀態與 SaveImage 結果，將紀錄存到 `test_results/workflow_result.json`。產圖位於 ComfyUI 的 `output/`，前綴 `mossimage_cpp_test`。單元測試使用替代程序驗證參數、圖片格式、錯誤與取消清理，不會載入 GPU 模型。

## 範圍與限制

- 舊整合節點支援文字生圖。獨立採樣器的 LoRA 與 latent 輸入限制見上文。
- `high` 搭配 INT8 DiT；`reference` 需要 bf16 DiT；`turbo` 需要上游支援的 INT4 / NVFP4 DiT。品質模式不會自動轉換權重。
- 預設 `min_memory`。執行前會卸載同一個 ComfyUI 程序內快取的模型，讓出 VRAM。其他程序占用的 VRAM 需自行釋放。
- 預設每張圖啟動一次引擎，完成或取消後回收程序與暫存檔；連續生圖可用 `keep_engine` 保留程序。取消會終止專屬的子程序。
- 此節點呼叫 CUDA 版 `moss-infer`；所需 GPU 與驅動以該套件為準。未驗證 Vulkan / HIP 引擎。
- 節點執行時不會下載模型或執行 shell；執行檔只能由本機安裝或環境變數指定。引擎輸出寫入專屬暫存目錄，再交由 ComfyUI 儲存。

獨立採樣器另外以 `tests/test_modular.py` 驗證原生 conditioning、patch 拒絕、共享記憶體身分檢查及 component 請求格式。與舊節點一樣，執行前卸載 PyTorch 模型快取，完成／失敗／取消後回收專屬 C++ 程序；不會常駐占用 GPU。

上游版本：2.2.1，原始碼 commit `f4f38f5e90d9990795d125b357574a68aadbe924`。上游二進位套件的授權與 CUDA notices 保留在 `runtime/` 內。

### GPU LoKr 合併

LoKr 快取建立已加入 GPU 分塊合併，無 CUDA 或 GPU 失敗時沿用 CPU。單組快取與磁碟空間需求不變。本機完整模型量測由約 51 秒降至 18 秒，輸出權重逐位元相同；詳見 [GPU_LOKR_REPORT.md](GPU_LOKR_REPORT.md)。


### DoRA LoKr 快取

已支援一般 LoKr 與 DoRA LoKr 組合的合併快取；首次建立逐層使用 ComfyUI 原生計算並由 GPU 重新量化，命中後重用完整模型。首次建立可能額外需要約 20 GB 以上差值暫存空間，詳見 [DORA_CACHE_REPORT.md](DORA_CACHE_REPORT.md)。


## 參考專案

| 專案 | 用途 |
| --- | --- |
| [mossimage.cpp / MossInfer](https://github.com/mossimage-cpp/mossimage.cpp)（Apache-2.0） | C++ 推論引擎與 component session 協定；`patches/` 是對 2.2.1（`f4f38f5`）的修改，LoKr helper 直接編譯其合併與量化函式 |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | 節點 API、原生採樣迴圈、LoRA／WeightAdapter、SkipLayerGuidance 與 ReferenceLatent 的行為依據 |
| [OpenMOSS MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | 模型架構、設定檔與 tokenizer；參考圖位置排列沿用 Moss-Image-Edit 的 `index_timestep_zero_text_start` |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Power LoRA Loader 相容性與效能比對 |
| [Dao-AILab fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)、[FlashAttention](https://arxiv.org/abs/2205.14135) | 數學等價加速研究的參考 |
| [NVIDIA CUDA 文件](https://docs.nvidia.com/cuda/) | cuBLAS 可重現性、浮點與非同步傳輸的依據 |

`patches/` 內的檔案為 mossimage.cpp 的衍生修改，依 Apache-2.0 散布；其餘程式碼另行授權前請勿假設任何授權。
