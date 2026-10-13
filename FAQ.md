# 常見問題(FAQ)

## 環境問題

### Q1:`pip install` 失敗怎麼辦?
先確認 Python 版本(建議 3.10 以上):

```bash
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

仍失敗請開 [Issue(提問)](https://github.com/ndhuccc/idl-2026-fall/issues/new/choose),並貼上**完整錯誤訊息**與作業系統。

### Q2:沒有 GPU 可以完成作業嗎?
可以。所有作業皆設計為 CPU 可完成;若需要 GPU,可使用 Google Colab 或 Codespaces。

## 繳交問題

### Q3:自動評分沒有通過,但我本機是通過的?
常見原因:
1. 使用了絕對路徑
2. 套件版本與 `requirements.txt` 不同
3. 隨機種子未固定

### Q4:可以遲交嗎?
見 [`syllabus.md`](syllabus.md) 的遲交規範;特殊情況請使用 Issue「申請」範本。

## 其他

(學期中持續補充)
