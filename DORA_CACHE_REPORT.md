# DoRA 合併快取支援

支援由一般 2D LoKr 與帶 DoRA 的 2D LoKr 組成的 MODEL patch。整個堆疊一起合併，保留各層原始順序、強度及 QKV 切片；不將 DoRA 與其他 LoKr 分開套用。其他不支援的 adapter 混用仍沿用原備援。

快取鍵由正規化原始 LoKr 因子、DoRA scale、強度、來源模型及 helper 雜湊建立。命中時不計算完整差值。首次建檔使用 ComfyUI 原生 calculate_weight，逐層在 ComfyUI 指定裝置執行 DoRA，再將完整差值分批匯出給 helper。helper 以 GPU 分塊反量化、加差值、重新量化，輸出單一合併模型；推論引擎不再接收這些差值檔案。

已更新 runtime/moss-lokr-cache.exe；Python 修改需重啟 ComfyUI。

驗證：
- 32 個單元測試通過，包含 LoKr/DoRA 正反順序、非單位／負強度、BF16 快取結果及命中後禁止重新計算。
- INT8 GPU 合併含完整差值與多個 LoKr：ConvRot 分組 4/16/64/256/1024，與強制 CPU 備援逐位元一致。
- 僅讀取使用者指定的兩份權重檔，各取 layers.0.proj.o 層；使用合成基底，兩種順序比對 GPU/CPU 原生 DoRA 計算，最大絕對差分別 5.22e-8、1.04e-7。沒有讀取工作流程或提示詞。

限制：尚未實測這兩份權重的完整模型建檔與出圖時間／RAM 峰值。首次建立的差值暫存是 FP32，大範圍 DoRA 可能另需約 20 GB 或更多磁碟空間；單批保留約 512 MiB 的 CPU 差值，另需單層基底／GPU 計算工作區。暫存由 TemporaryDirectory 在完成或失敗後清除，仍只保留最近一個最終快取。GPU OOM 或建檔失敗會記錄警告並回到原備援，可能再次遭遇原本的 host RAM 限制。完整差值的相減再相加及 GPU reduction 會有浮點誤差，不保證與原生出圖逐像素一致。
