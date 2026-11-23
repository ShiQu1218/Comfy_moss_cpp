# GPU LoKr 合併驗證

2026-11-15，本機 RTX 5080 16 GB、MOSS SFT ConvRot INT8。

新版 `runtime/moss-lokr-cache.exe` 在建立快取時，以每批 128 列在 GPU 執行反量化、ConvRot、LoKr 加總、旋轉與 INT8 重新量化。沿用引擎的 adapter 解析、因子展開及驗證；BF16 仍走原 CPU 路徑。分解因子的乘法仍在 CPU 執行，避免改變既有 double 累加順序。

| 完整模型建立合併檔（含啟動與磁碟讀寫） | 秒數 |
| --- | ---: |
| 原 CPU helper | 50.955 |
| GPU helper | 17.986 |

同一份約 595 MB LoKr、強度 1，輸出整份 5.4 GB safetensors 的 SHA-256 完全相同。此組約加速 2.83 倍、時間減少 64.7%。為單次定位量測，未清除 OS 檔案快取；CPU 測量期間也有小型編譯／測試活動，非嚴格隔離基準。原始數據：`test_results/gpu_merge/comparison.json`。

驗證：
- `tests/check_gpu_merge.py`：ConvRot 分組 4、16、64、256、1024；兩個分解 LoKr，正負強度，CPU/GPU 輸出逐位元相同，並確認 GPU 路徑實際執行。
- 強制 CPU 備援與原 helper 輸出逐位元相同。
- 既有 30 個單元測試全數通過，包含 BF16、Unicode、快取失效、取消及原始檔保護。

無 CUDA 裝置或 GPU 操作失敗會使用 CPU 合併。GPU 結果整層完成後才提交到主記憶體，失敗不會將部分合併結果再次套用。可用環境變數 `MOSS_LOKR_CPU=1` 強制 CPU。測試程式需要先保留原 helper 為 `native/build/moss-lokr-cache-cpu.exe`。

仍保留單組磁碟快取與原本的原子替換／鎖定，空間需求不變。helper 雜湊改變後，下一次自動重建。已安裝本機編譯版本，無須改 workflow；若沒有更改 Python 程式，現有 ComfyUI 下一次啟動 helper 就會使用新版。

建置：在已安裝 CUDA Toolkit、Visual Studio C++、CMake 與 Ninja 的 Windows 環境執行 `native/build_gpu.cmd`，產物位於 `native/build-gpu`。腳本依本機 GPU 建置；其他架構需自行指定 `CMAKE_CUDA_ARCHITECTURES` 後建置。

限制：本次沒有重跑完整出圖流程（以整份合併模型位元相同驗證權重），尚未量測顯示記憶體峰值、模擬 CUDA OOM 或驗證其他 GPU／Linux。分塊 FP32 工作區有固定列數上限，但因子與 CUDA context 仍占用額外記憶體。尚未實作直接將 GPU 合併結果交給常駐推論引擎，仍需完整模型磁碟寫入與後續載入。
