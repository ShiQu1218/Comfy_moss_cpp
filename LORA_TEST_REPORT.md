# 原生 LoRA 連線驗證

此為第一版一般 LoRA 驗證；後續已新增 LoHa／LoKr、DoRA／LoCon 等轉換及原生 SLG，最新結果見 [EXTENDED_TEST_REPORT.md](EXTENDED_TEST_REPORT.md)。

2026-11-15，RTX 5080 16 GB，ComfyUI 0.44.2，MossInfer 2.2.1 CUDA 13。

沿用原生 `Load LoRA`／`Load LoRA (Model Only)` 的 MODEL patch，將一般 LoRA 張量、alpha 與原強度轉送給 C++ component session；沒有新增選檔 UI、修改基礎模型或重複套用已編碼的 CLIP LoRA。

## 單元測試

新增 3 項測試，以 `comfy.lora.calculate_weight` 作為數值基準，驗證正／負強度、同目標多重堆疊、alpha／無 alpha／alpha 0、fused QKV 的 Q/K/V 切片、共用張量、diff bias、原始張量不被修改，以及不支援的 patch／DoRA 會明確失敗。既有 component 測試另驗證 LoRA 檔案與負強度正確傳入 `run_begin`。

## GPU 工作流程

使用既有 SFT INT8 DiT、INT8 Qwen3.5 文字編碼器、FLUX.2 small VAE 與 `moss_黑白線稿.safetensors`。512×512、12 步、ER-SDE／simple、CFG 4、seed 42，提示詞沿用紅蘋果測試。

| 情境 | 結果 | workflow 總耗時 |
| --- | --- | ---: |
| 未使用 LoRA | 成功 | 10.584 秒 |
| 原生 Load LoRA，model 強度 0.7 | 成功 | 12.358 秒 |
| 原生 Load LoRA，強度 0 | 與未使用 LoRA 的 RGB 像素完全相同 | 8.463 秒 |
| 再串接 Model Only，同一 LoRA 強度 -0.2 | 成功，結果與 0.7 時不同 | 8.882 秒 |

強度 0.7 時引擎回報 252 個 adapter；串接時 504 個 adapter，皆為 side branch，unknown 與 warnings 都為空。匯出檔案會依 patch 堆疊位置與強度分組，因此不一定與原始 LoRA 檔案數相同。

已目視確認套用前／後圖片正常。原始 history 與事件位於 `test_results/lora/{off,on,zero,chain}_1.json`，像素比較位於 `verification.json`，引擎 adapter 回報保留在 `server.stderr.log`。這是功能驗證，各輪快取狀態不同，不能拿耗時表推算 LoRA 額外成本。

範例：[lora.json](workflows/lora.json)、[lora_api.json](workflows/lora_api.json)。完成後沒有殘留 moss-server 程序，暫存 LoRA 在執行結束時移除。

尚未實測：含 CLIP 權重的 LoRA、其他 GPU、reference／turbo；目前不支援 DoRA、LoCon、LoHa／LoKr、reshape 與非標準 MODEL patch。原有 batch／ControlNet／SLG 等限制不變。
