# LoKr cache helper

此工具將節點匯出的正規化 LoKr 或完整差值合併到來源 DiT 的副本。只接受新輸出檔案，支援目前模型使用的 canonical tensor names、ConvRot INT8 與 BF16。ConvRot INT8 優先使用 GPU，失敗時退回 CPU。Python 呼叫端負責鎖定、快取辨識、暫存檔、取消與原子替換。遇到不相容模型會退回原本的引擎合併流程。

重用 MossInfer 2.2.1 的 `lora_merge.cpp`、LoKr parser 與量化 codec，固定 commit `f4f38f5e90d9990795d125b357574a68aadbe924`。`normalized_metadata.cpp` 只提供正規化輸入會使用的 metadata／alpha 函式，拒絕非空 metadata 與一般 low-rank merge，以免連入完整的模型轉換器。建置時停用 FMA contraction 與 fast-math。

在啟動 ComfyUI 前設定 `MOSS_LOKR_FAST_ROTATION=1`，可試用 g=256 的四階段 Hadamard 旋轉。預設使用原本的密集旋轉。啟用或移除環境變數會重建最近一組 LoKr 快取。快速旋轉會改變浮點累加順序，可能改變 INT8 權重和量化比例。GPU 失敗時仍使用原本的 CPU 備援。

RTX 5080 上以同一份 SFT INT8 模型及 LoKr 各交錯量測三次，含磁碟讀寫的建檔中位數由 18.36 秒降至 12.89 秒（約少 30%）。兩組耗時範圍重疊，且 GPU 有其他記憶體占用。此量測涵蓋快取建檔。

Windows 需要 Visual Studio 的 MSVC x64 C++ 工具、CUDA Toolkit（含 `nvcc`）、CMake 3.28+ 與 Ninja。請在專案根目錄取得上游原始碼，核對 commit 後再建置。

```powershell
git clone --depth 1 --branch v2.2.1 https://github.com/mossimage-cpp/mossimage.cpp .upstream
git -C .upstream rev-parse HEAD
.\native\build_gpu.cmd
Copy-Item native/build-gpu/moss-lokr-cache.exe runtime/moss-lokr-cache.exe
Copy-Item .upstream/LICENSE,.upstream/NOTICE,.upstream/THIRD_PARTY_NOTICES.md runtime/
```

若已有 `.upstream`，直接核對其 commit 即可。Windows 腳本會載入 MSVC x64 環境並針對本機 GPU 編譯。Linux 可用 CUDA Toolkit 支援的 C++20 主機編譯器執行 `cmake -S native -B native/build-gpu -G Ninja -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release`，接著用 `cmake --build native/build-gpu` 建置，並將不含 `.exe` 的產物複製到 `runtime/`。跨機器建置時，將 `native` 換成目標 GPU 的 compute capability。
