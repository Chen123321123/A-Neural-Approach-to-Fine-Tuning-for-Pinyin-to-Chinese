# Pinyin → Chinese (mT5) — Fine-tuning Project

Neural IME-style converter: fine-tunes T5-family encoder–decoder models (mT5 / Mengzi-T5) to map **Pinyin sequences** to **Chinese characters**, with optional **typo-robust** training.

## What this repo does
- **Data prep**: cleans/normalizes Pinyin (tone/spacing, ü→v, retroflex fixes), aligns Pinyin–Han pairs, and can **inject realistic typos** (near-key substitution, deletion, swap) to improve robustness. :contentReference[oaicite:0]{index=0}
- **Modeling**: fine-tunes **mT5 (small/base/large)** and **Mengzi-T5-base** for seq2seq generation; inputs are prompted as  
  `Convert Pinyin to Chinese: <pinyin>`; decoding uses **beam search (beam=4)**.
- **Training**: Hugging Face `Seq2SeqTrainer` + **AdamW**, **lr=3e-6**, **epochs=20**, **batch=16**, **bf16** mixed precision; metrics: **CharAcc, ChrF, BLEU, ROUGE**.
- **Inference**: `generate(max_length=64)` then post-process to text.

## Key results (from the report)
- On clean data, **mT5-large** achieves **CharAcc 70.15%** and **ChrF 53.59%** (Table 1).  
- With typo-augmented training, mT5-large improves on **typo test** to **CharAcc 44.20% vs 38.73%** (Table 3).  
- Loss curves show **model scale matters**—mT5-large converges best (Figure 3). :contentReference[oaicite:1]{index=1}
