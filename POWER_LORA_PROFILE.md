# Power LoRA Loader 等待時間調查

2026-11-15，本機 RTX 5080、INT8 MOSS SFT、high／min_memory。測試沿用相同中性紅蘋果提示詞、256×256、4 步、ER-SDE／simple、CFG 4、seed 42；比較相同 LoKr／LoRA 經 rgthree Power Loader 與原生 Loader 的行為。

| 流程 | C++ run_begin：載入與合併 | 全流程 |
| --- | ---: | ---: |
| 無 LoRA Loader | 2.395 秒 | 8.314 秒 |
| Power Loader，1 個項目關閉 | 2.453 秒 | 6.628 秒 |
| Power Loader，18 個項目全關閉 | 2.685 秒 | 7.822 秒 |
| Power Loader，1 個一般 LoRA | 2.919 秒 | 8.604 秒 |
| 原生 Loader，同一個一般 LoRA | 2.758 秒 | 7.928 秒 |
| Power Loader，1 個約 595 MB 的 LoKr | 50.125 秒 | 54.187 秒 |
| 原生 Loader，同一個 LoKr | 50.278 秒 | 55.103 秒 |

LoKr 的匯出分別只花 0.469／0.295 秒，server 初始化約 0.2 秒，首次評估約 0.17 秒。等待集中在 C++ 合併權重與 INT8 重新量化。rgthree 在本機版本直接呼叫原生 LoraLoader，關閉的項目不會載入；本對照未重現單純使用 Power Loader 就固定增加 40 秒的現象。

像素比對：兩種 Loader 的同一 LoKr 結果完全相同，同一一般 LoRA 也完全相同；Power Loader 全關閉（1 個或 18 個項目）與無 Loader 的 RGB 像素完全相同。各輪快取狀態不同，本表用來定位主要階段，不是完整統計效能基準。

本次修正讓進度條在模型準備前先顯示 0%，約在採樣器執行後 5～7 ms 即收到進度事件；並加入各準備階段耗時。7 個 GPU 工作流程均完成 4 步，首次評估只記錄一次，原始事件與彙整保留於 `test_results/power_lora/`。

本表記錄加入合併快取之前的行為：每張圖啟動新的 C++ 引擎，相同 LoKr 下一張仍重新合併；進度顯示修正沒有消除成本。後續已新增磁碟快取，見 [LOKR_CACHE_REPORT.md](LOKR_CACHE_REPORT.md)。LoKr 並非在所有引擎、模型精度或作用層數下都固定需要 50 秒；其他 adapter／GPU／精度未在此比較。
