# 🎧 音声対話AIアシスタント（YouTubeライブ連携）

![デモ](/assets/images/voicebot_demo.jpg)

## 概要
YouTubeライブのチャット取得・音声認識・翻訳・LLM応答・音声合成を統合した、
**リアルタイム音声対話ボット**のプロトタイプ。

## 使用技術
- Python, Google Colab  
- pytchat, Whisper, （任意の）LLM API, DeepL API, VoiceVox Core  
- 非同期処理（asyncio, nest_asyncio）

## 処理フロー
音声入力 → Whisper（ASR） → 翻訳 → LLM応答生成 → VoiceVox（TTS） → 出力

## レイテンシ内訳（例）
| 処理 | 時間(ms) |
|------|---------|
| ASR(Whisper) | 400 |
| 翻訳 | 100 |
| 応答生成(LLM) | 800 |
| 音声合成(TTS) | 300 |
| **合計** | **1600** |

## 今後の改善
- 非同期バッチ化・キャッシュで待ち時間を短縮
- 音声感情推定→演出（色調・効果音）への反映