# 擴充 LoRA 格式與跳層引導驗證

2026-11-15，RTX 5080 16 GB、ComfyUI 0.44.2、PyTorch 2.10.0+cu128、MossInfer 2.2.1 CUDA 13。沿用既有 INT8 SFT DiT、INT8 Qwen3.5 文字編碼器與 FLUX.2 small VAE。

## 數值與介面驗證

- LoHa／LoKr 保留原因子與 alpha，重新載入匯出檔後與 `comfy.lora.calculate_weight` 的結果比較，包含未分解 LoKr 忽略 alpha 的規則。
- DoRA、LoCon、reshape、Tucker LoHa、Tucker／高維 LoKr 透過原生計算轉成完整差值，與原生完整堆疊比較；含正／負強度、前置一般 LoRA，以及 pristine backup 的選取，原始權重不會修改。
- 無效原生 adapter 即使只記錄錯誤而未丟出例外，也會被偵測並停止轉送；會拒絕改變參數形狀或非有限的結果。
- 直接使用本機的原生 Simple／完整 SLG 節點建立 callback，驗證辨識、區間端點、CFG 1、多個完整 SLG、重縮放及原生公式一致。未知 callback 不會被當成 SLG 忽略。
- component 測試驗證作用區間內／外選取不同 prediction rows，並檢查 C++ 確實完成請求的 rows；原有共享記憶體、取消、採樣及 LoRA 測試保留。

共 19 項單元測試：`test_nodes.py` 4、`test_modular.py` 6、`test_lora.py` 6、`test_guidance.py` 3。

## GPU 工作流程

通過 11 個實際 ComfyUI API 流程，原始事件與 history 保留在 `test_results/extended/`，圖片及像素比較彙整於 `verification.json`。

| 流程 | 結果 |
| --- | --- |
| 未加 LoRA／SLG 的基準 | 成功 |
| 原生 Simple、完整 SLG（含重縮放）、兩者串接 | 全部成功，圖片與基準不同 |
| LoHa、LoKr、DoRA、LoCon | 全部成功，圖片與基準不同 |
| SLG 不在作用區間內 | RGB 像素與基準完全一致 |
| 完整 SLG scale 0 | RGB 像素與基準完全一致 |
| 原生 Load LoRA → Simple → 完整 SLG → C++ 採樣器 | 1024×1024、50 步、ER-SDE／simple、CFG 4 成功 |

前十個流程為 256×256、4 步、seed 42。格式測試使用固定種子生成、與 MOSS 單一線性層維度相符的測試 adapter，經原生 `LoraLoaderModelOnly` 載入，並非宣稱驗證了所有社群 LoRA 檔案。DoRA／LoCon 的 GPU 測試也驗證量化 MODEL 取出與完整差值合併路徑。

四份小型測試 adapter 保留於本機 `ComfyUI/models/loras/ComfyMossCppTests/`，供重現格式測試使用；正式範例 workflow 不依賴它們。測試用的 8189 伺服器已關閉，未修改原本的 ComfyUI 啟動參數。

最後的 1024×1024 流程使用既有 `moss_黑白線稿.safetensors`，Simple 跳過第 22 層（0～50%），完整 SLG 跳過第 23 層（0～30%）、scale 1.5、rescaling 0.2。單次 workflow 約 65.5 秒，已目視確認圖片正常；這是功能驗證，快取狀態不同，不能作為與舊版的效能比較。

範例：[slg.json](workflows/slg.json)、[slg_api.json](workflows/slg_api.json)。重啟主 ComfyUI 後即可沿用原生 LoRA 與 SLG 節點接線。

## 限制

完整差值轉換較耗 RAM、暫存空間與載入時間；INT8 合併可能增加重新量化誤差。仍只支援 MOSS-Image 的相同參數形狀、batch 1、單組 conditioning；ControlNet、遮罩、參考圖與第三方任意 callback 不在本次範圍。未測 Linux、其他 GPU、reference／turbo、所有社群 adapter，以及所有其他 WeightAdapter 的 GPU 路徑。
