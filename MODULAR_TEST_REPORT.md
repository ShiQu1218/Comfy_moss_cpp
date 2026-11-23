# 獨立採樣器驗證（2026-11-15）

LoRA 後續已新增支援，驗證與目前格式限制見 [LORA_TEST_REPORT.md](LORA_TEST_REPORT.md)。以下保留首次獨立採樣器驗證結果。

`KSampler (MOSS Image C++)` 使用原生 MODEL、正／負 CONDITIONING 與 LATENT，輸出標準 LATENT。範例見 `workflows/modular.json` 與 `modular_api.json`；僅替換 KSampler，其餘為原生 UNETLoader、CLIPLoader、CLIPTextEncode、EmptyFlux2LatentImage、VAELoader、VAEDecode、SaveImage。

- RTX 5080 16 GB、ComfyUI 0.44.2、PyTorch 2.10.0+cu128、MossInfer 2.2.1 CUDA 13，既有 INT8 SFT DiT、INT8 Qwen3.5 文字編碼器與 FLUX.2 small VAE。
- 256×256、3 步的完整連線測試成功。
- 1024×1024、50 步、ER-SDE／simple、CFG 4、seed 42 的完整連線測試成功；總耗時 23.537 秒，採樣節點含模型準備與程序管理約 21.133 秒。這是單次功能驗證，沒有重做完整效能比較。
- 已目視確認產圖正常：`D:/ComfyUI/ComfyUI_windows_portable/ComfyUI/output/moss_cpp_modular_00001_.png`。
- 執行事件與 history 位於 `test_results/modular/full_1.json`、`smoke_1.json`。生成完沒有殘留 moss-server 子程序。
- 新增 5 項單元測試：原生 conditioning 與 attention mask、拒絕不支援的 MODEL patch、共享記憶體身分／token 檢查、component 請求格式（允許 ER-SDE 評估表列以外的 sigma）、取消後回收程序與 mapping。原有整合節點的 4 項測試也通過。

目前支援單張 batch、單組正／負 conditioning，不支援 LoRA、SLG、ControlNet、參考圖、noise mask、區域／時間排程等；明確拒絕，避免忽略原 workflow 設定。Linux、其他 GPU 與圖生圖未做實機驗證。主 ComfyUI 需重新啟動才能載入新節點。
