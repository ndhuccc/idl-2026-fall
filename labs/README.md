# 實驗(Labs)

每個實驗一個資料夾,結構固定:

```text
labN/
├── README.md      # 題目說明與評分表(教師撰寫)
├── starter/       # 學生起始碼(公開)
├── solution/      # 參考解答(截止後才公開,或放私有 Repo)
└── tests/         # 自動測試(pytest),亦用於 GitHub Actions 自動評分
```

## 執行測試

```bash
# 學生:測試自己的實作(預設測試 starter/)
pytest labs/lab01/tests -v

# 教師:驗證解答是否通過全部測試
LAB_TARGET=solution pytest labs/lab01/tests -v
```

## 新增實驗

```bash
cp -r labs/lab01 labs/lab02
# 修改 lab02 內的程式碼、測試與 README
```

> 建議每次新增實驗後,同時確認「solution 全過、starter 全部失敗」,以證明測試有鑑別力。
