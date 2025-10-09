# Overview

```mermaid
---
config:
      theme: redux
---
flowchart TD
    A(["🏎️ Odometry<br/>オドメトリ<br/><sub>位置と速度の推定</sub>"])
    B(["🧭 Trajectory Planning<br/>経路計画<br/><sub>目標までの経路生成</sub>"])
    C(["⚙️ Control<br/>制御<br/><sub>モータ出力と姿勢制御</sub>"])
    D(["👀 Detection<br/>検出<br/><sub>物体・ボトル検出</sub>"])
    E(["🎙️ Voice / Activation<br/>音声・起動<br/><sub>ユーザー入力</sub>"])

    A -- "🔗 Pose / Velocity<br/>位置・速度データ" --> B
    B -- "🔗 Path Commands<br/>経路コマンド" --> C
    D -- "🔗 Object Info<br/>物体情報" --> B
    E -- "🔗 Activation Signal<br/>起動シグナル" --> D
    D -- "🔗 Feedback<br/>検出結果" --> C
```
