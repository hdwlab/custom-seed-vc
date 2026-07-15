# SocketIOを用いたリアルタイム音声変換

本Exampleでは、SocketIOを用いた、リアルタイム音声変換のサーバーとクライアントの利用方法を説明します。

## 環境構築

SocketIO用のモジュールをインストールしてください。
Client側のマシンでは、`--only-group`オプションを使うことで必要最小限のモジュールのみをインストールできます。

```bash
$ cd /path/to/custom-seed-vc/

# Server側でのモジュールをインストール
$ uv sync

# Client側でのモジュールをインストール
$ uv sync --only-group client
```

## 使い方

### Serverの起動

Serverを起動するには、以下のコマンドを実行します。
起動には時間がかかる場合があります。

```bash
$ cd /path/to/custom-seed-vc/
$ uv run python seed_vc/socketio/server.py
```

main repo 側で作成した `presets/` を使う場合は `--presets-dir` を指定します。SeedVC 用の vector space は main repo の speaker-space builder で `spaces/seed_vc.npz` として作成してください。

```bash
$ uv run python seed_vc/socketio/server.py --presets-dir /path/to/presets
```

起動が完了すると次のようなメッセージが表示されます。
```bash
$ uv run python seed_vc/socketio/server.py
[15:09:22] [SERVER] [INFO] 🚀 Starting server imports...
[15:09:22] [SERVER] [INFO] ⏳ Importing seed_vc modules (this may take a while)...
[15:09:37] [SERVER] [INFO] ✅ All imports completed!
[15:09:37] [SERVER] [INFO] 🎙️  Starting voice conversion server on 0.0.0.0:5000 ...
[15:09:37] [SERVER] [INFO] 🔄 Initializing global VoiceConverter...
[15:09:53] [SERVER] [INFO] ✅ Global VoiceConverter ready!
[15:09:53] [SERVER] [INFO] 🌟 Ready to accept connections!
```

### Clientの起動

Clientを起動するには、以下のコマンドを実行します。
Serverが起動していることを確認してから実行してください。

```bash
$ cd /path/to/custom-seed-vc/
$ uv run python seed_vc/socketio/client.py
```

operator ごとの voice preset を使う場合は `--operator-id` を指定します。`gender` 付き preset を用意している場合だけ `--gender female` / `--gender male` で候補を絞れます。

```bash
$ uv run python seed_vc/socketio/client.py --operator-id alice
```

起動が完了して、Serverと接続できると次のようなメッセージが表示されます。

```bash
$ uv run python seed_vc/socketio/client.py
[15:11:57] [CLIENT] [INFO] 🔗 Connecting to http://localhost:5000
[15:11:57] [CLIENT] [INFO] 🔗 Connected to server
[15:11:57] [CLIENT] [INFO] 🎧 Streaming... (Ctrl+C to stop)
```

この状態で、Client側のマイクから音声を入力すると、Server側で音声変換が行われ、変換された音声がClient側のスピーカーから再生されます。

Clientは起動時点のOS / `sounddevice` のデフォルト入力デバイスとデフォルト出力デバイスを使用します。
利用可能なdeviceは `--list-devices` で確認できます。
マイクとスピーカーはdevice indexまたは名前で指定できます。

```bash
$ uv run python seed_vc/socketio/client.py --list-devices
$ uv run python seed_vc/socketio/client.py \
    --input-device 3 \
    --output-device "USB Headset"
```

指定したdeviceが44100 Hz、mono、int16形式に対応していない場合は、接続前にerrorを表示して終了します。

ServerとClient間にラグがあると感じる場合は一度、Client側だけを再起動することで改善される場合があります。

リアルタイム変換中にServer側でエラーが発生した場合は、原音をそのまま返さず、該当チャンクを無音に置き換えます。
Clientには `conversion_error` イベントが通知され、エラー内容がログに表示されます。
原音を出力する場合は、障害時のfallbackではなく `passthrough` modeを明示的に設定してください。

Serverの監視には次のendpointを利用できます。

- `GET /health/live`: ASGI processの死活確認
- `GET /health/ready`: engine初期化状態、動作mode、接続数の確認
- `GET /metrics`: 接続数、変換error数、破棄chunk数、処理時間、RTFのPrometheus形式metrics

### FastAPIによる設定の変更

Serverを起動した状態でAPIを叩くと、音声変換の設定を変更することができます。
次のようなAPIが用意されています。

- 変換モードの変更
    - convert : 音声変換を行うモード（デフォルト）
    - passthrough : 音声変換を行わず、入力音声をそのまま出力するモード
    - silence : 音声を無音にするモード
```bash
$ curl -X POST "http://localhost:5000/api/v1/mode" \
    -H "Content-Type: application/json" \
    -d '{"mode": "passthrough"}'
```

- リファレンス音声の変更
```bash
$ curl -X POST "http://localhost:5000/api/v1/reference" \
    -H "Content-Type: application/json" \
    -d '{"file_path": "assets/examples/reference/trump_0.wav"}'

# セキュリティの関係からデフォルトでは assets/examples/reference/ 以下の音声ファイルのみを指定できます。
# 別のディレクトリの音声ファイルを指定したい場合は、server.pyを起動する際に、`--allowed-audio-dirs`オプションをつけて起動してください。
$ uv run python seed_vc/socketio/server.py --allowed-audio-dirs /path/to/your/audio/dir
```

- Voice Preset の一覧
```bash
$ curl "http://localhost:5000/api/v1/presets"
```

- 音声変換モデルの各種パラメータの変更
```bash
$ curl -X POST "http://localhost:5000/api/v1/parameters" \
    -H "Content-Type: application/json" \
    -d '{"block_time": 0.18,"extra_time_ce": 0.5}'
```

- 音声変換モデルの再読み込み
```bash
# デフォルトの音声変換モデルを再読み込み
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{}'

# ファインチューニングした音声変換モデルを読み込み
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{"checkpoint_path": "examples/fine-tuning/runs/my_run/ft_model.pth", "config_path": "examples/fine-tuning/runs/my_run/config_dit_mel_seed_uvit_xlsr_tiny.yml"}'
```

- オフラインのファイル変換（Clientが接続していない状態でのみ実行可能）
```bash
# Server上のファイルパスを指定して変換
$ curl -X POST "http://localhost:5000/api/v1/convert" \
    -H "Content-Type: application/json" \
    -d '{"input_path": "assets/examples/reference/trump_0.wav", "output_path": "assets/examples/reference/converted.wav"}'

# ファイルをアップロードして変換結果をダウンロード (リファレンス音声の一時指定も可能)
$ curl -X POST "http://localhost:5000/api/v1/convert/upload" \
    -F "input_file=@/path/to/input.wav" \
    -F "reference_file=@/path/to/reference.wav" \
    -o converted.wav
```

複数のServer側ファイルを変換完了を待たずに投入する場合は、非同期ジョブAPIを使用できます。
1ジョブは1件から100件までで、単一workerが投入順に処理します。

```bash
$ curl -X POST "http://localhost:5000/api/v1/convert/jobs" \
    -H "Content-Type: application/json" \
    -d '{
      "items": [
        {"input_path":"assets/examples/reference/trump_0.wav","output_path":"assets/examples/reference/output-1.wav"},
        {"input_path":"assets/examples/reference/trump_0.wav","output_path":"assets/examples/reference/output-2.wav"}
      ]
    }'

$ curl "http://localhost:5000/api/v1/convert/jobs/<job-id>"
```

ジョブの状態は `queued`、`running`、`succeeded`、`failed` のいずれかです。
一部itemが失敗しても後続itemは処理され、結果は `result.items` に記録されます。
先頭ジョブの受理からキューが空になるまではリアルタイム変換と同期オフライン変換を受け付けません。
ジョブ情報はメモリ上に直近100件まで保持され、Server再起動時に失われます。
未完了ジョブが100件に達した場合はHTTP 429で拒否されます。

Serverを `--presets-dir` 付きで起動すると、リアルタイム変換と同じ決定規則でoperatorごとのvoiceを選択できます。
変換中だけ選択したpresetとspeaker vectorが適用され、変換後は元のreference音声に戻ります。

```bash
# Server側のfile pathを使う場合
$ curl -X POST "http://localhost:5000/api/v1/convert" \
    -H "Content-Type: application/json" \
    -d '{"input_path":"assets/examples/reference/trump_0.wav","output_path":"assets/examples/reference/output.wav","operator_id":"alice","gender":"female"}'

# uploadする場合
$ curl -X POST "http://localhost:5000/api/v1/convert/upload" \
    -F "input_file=@/path/to/input.wav" \
    -F "operator_id=alice" \
    -F "preset_id=female_001" \
    -o converted.wav
```

`operator_id`だけを指定した場合は全presetが候補になります。
`gender`または`preset_id`を使う場合は `operator_id`も必要です。
upload APIでは `reference_file`と`operator_id`を同時に指定できません。
選択されたpreset IDはpath指定APIのJSONでは `preset_id`、upload APIでは `X-Voice-Preset-ID` response headerに入ります。
versioned preset bundleではbundle IDも `preset_bundle_id` または `X-Voice-Preset-Bundle-ID` に入ります。

Server起動後にbundleを更新した場合は、Clientとoffline変換が動いていない状態で検証付きreloadを実行できます。
新しいbundleの読み込みやSeed-VC互換性検証に失敗した場合は、現在のbundleがそのまま維持されます。

```bash
$ curl -X POST "http://localhost:5000/api/v1/presets/reload"
```

詳しいAPIの仕様は、Serverを起動した状態で、ブラウザから`http://localhost:5000/docs`にアクセスすることで確認できます。

## Dockerを用いた実行

Dockerを用いてServerを実行することもできます。以下の手順で実行できます。

### Dockerイメージのビルド

```bash
$ export COMPOSE_FILE=docker/socketio/docker-compose.yml
$ docker compose build
```

### Dockerコンテナの起動・停止

```bash
# コンテナの起動
# デフォルトではポート5000で起動
$ docker compose up -d

# コンテナの停止
$ docker compose down
```

コンテナの起動後、通常と同じようにClientを起動してコンテナ上のサーバーに接続することでリアルタイム音声変換を行うことができます。
