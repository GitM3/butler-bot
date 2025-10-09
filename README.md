<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/f0d98dc6-efdd-4ed9-a299-fe6bd86a9c44" />

# 🤖 Butler-Bot 🍺

> 「人間より早くビールをキャッチする日も近いかも！」 —
> 我らの夢のロボット、**Butler-Bot**

ようこそ！このリポジトリは、居酒屋でビールをキャッチするロボット「Butler-Bot」の開発プロジェクトです。

---

## 📋 プロジェクトボード

作業の進行状況やタスクは、以下のプロジェクトボードで管理します。 👉
[プロジェクトボードはこちら](https://github.com/users/GitM3/projects/5)

---

## 🧭 開発フロー

```mermaid
flowchart LR
    A[📝 自分のタスクを Issue に登録] --> B[📊 Kanban で進行を管理]
    B --> C[💬 Commit メッセージに Issue を参照 (#番号)]
    C --> D[🚀 コードレビュー・マージ！]
```

## 🗂️推奨フォルダ構成

```bash
butler-bot/
├── README.md
├── members/
│   ├── zander/            # 各メンバーの作業用フォルダ
│   ├── sasaki/
│   └── ../
└── system/                # チームで統合する「本体」コード
    ├── main/
    ├── vision/
    ├── control/
    └── communication/
```

🧠 コミットメッセージ例

```bash
git commit -m "Add bottle detection model (#12)" <--- Please reference issue number
```

💬 "失敗しても、ビールは冷たいし、"

Let's code, and toast to progress! 🍻

(酒は飲まない人：お茶でも構わない)
