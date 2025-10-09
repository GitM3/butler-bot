<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/f0d98dc6-efdd-4ed9-a299-fe6bd86a9c44" />

# 🤖 Butler-Bot 🍺

ようこそ！このリポジトリは、居酒屋でビールをキャッチする
ロボット「Butler-Bot」の開発プロジェクトです。

## 📋 プロジェクトボード

- ロジェクトボード 👉 [こちら](https://github.com/users/GitM3/projects/5)
- System design here: 👉[こちら](./system/Docs/System_design.md)

## 🧭 開発フロー

    1) 📝 タスクを Issue に登録
    2) 📊 Kanban で進行を管理
    3) 💬 Commit メッセージで Issue 番号を参照 (#123 など)
    4) 🚀 コードレビューとマージ

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
