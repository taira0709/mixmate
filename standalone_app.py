#!/usr/bin/env python3
"""
MixMate Standalone Desktop Application
スタンドアロン版MixMateのエントリーポイント
"""

import os
import sys
import threading
import time
import webview
from pathlib import Path

# アプリケーションのベースディレクトリを設定
if getattr(sys, 'frozen', False):
    # PyInstallerでビルドされた場合
    BASE_DIR = Path(sys._MEIPASS)
else:
    # 開発環境の場合
    BASE_DIR = Path(__file__).parent

# static/templatesディレクトリのパスを設定
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'

# 既存のFlaskアプリをインポート
from app import app

# WebView API class
class Api:
    def __init__(self, window):
        self.window = window

    def save_file_with_dialog(self, data_b64, filename):
        """ファイルダイアログを使ってBase64データを保存"""
        try:
            print(f"[DEBUG API] save_file_with_dialog 呼び出し - ファイル名: {filename}")
            import base64
            import os

            # Base64デコード
            print(f"[DEBUG API] Base64データ長: {len(data_b64)} 文字")
            audio_data = base64.b64decode(data_b64)
            print(f"[DEBUG API] デコード後データサイズ: {len(audio_data)} bytes")

            # ファイル保存ダイアログを表示
            result = self.window.create_file_dialog(
                webview.SAVE_DIALOG,
                directory=os.path.expanduser("~/Desktop"),
                save_filename=filename,
                file_types=('WAV Files (*.wav)',)
            )

            print(f"[DEBUG API] ファイルダイアログ結果: {result}")

            if result and len(result) > 0:
                save_path = result[0]
                print(f"[DEBUG API] 保存パス: {save_path}")

                # ファイルを保存
                with open(save_path, 'wb') as f:
                    f.write(audio_data)

                print(f"[DEBUG API] ファイル保存完了: {save_path}")
                return {'status': 'success', 'path': save_path}
            else:
                print("[DEBUG API] ユーザーがファイルダイアログをキャンセル")
                return {'status': 'cancelled', 'message': 'ユーザーがキャンセルしました'}

        except Exception as e:
            print(f"[DEBUG API] エラー発生: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def save_file_to_desktop(self, data_b64, filename):
        """デスクトップに直接保存（フォールバック用）"""
        try:
            print(f"[DEBUG API] save_file_to_desktop 呼び出し - ファイル名: {filename}")
            import base64
            import os
            from datetime import datetime

            # Base64デコード
            audio_data = base64.b64decode(data_b64)

            # デスクトップパスを取得
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_path = os.path.join(desktop, filename)

            # ファイルを保存
            with open(save_path, 'wb') as f:
                f.write(audio_data)

            print(f"[DEBUG API] ファイル保存完了: {save_path}")
            return {'status': 'success', 'path': save_path}

        except Exception as e:
            print(f"[DEBUG API] エラー発生: {str(e)}")
            return {'status': 'error', 'message': str(e)}

def find_free_port():
    """利用可能なポートを見つける"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def _run_flask(port):
    """Flaskサーバーをバックグラウンドで起動"""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Flaskのメッセージを非表示

    app.run(
        host='127.0.0.1',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )

def main():
    """メイン実行関数"""
    # PyWebViewの設定：ダウンロードを有効にし、大きなファイル用に制限を緩和
    webview.settings['ALLOW_DOWNLOADS'] = True
    webview.settings['ALLOW_FILE_URLS'] = True

    # メモリとアップロード制限を緩和
    import sys
    import gc
    sys.setrecursionlimit(10000)  # 再帰制限をさらに増加
    sys.set_int_max_str_digits(0)  # 文字列制限解除

    # ガベージコレクションの調整（大きなファイル用）
    gc.set_threshold(100, 5, 5)  # より頻繁なガベージコレクション

    # 利用可能なポートを取得
    port = find_free_port()

    # Flaskサーバーをバックグラウンドで起動
    flask_thread = threading.Thread(target=_run_flask, args=(port,))
    flask_thread.daemon = True
    flask_thread.start()

    # 少し待ってからWebViewを起動
    time.sleep(1.5)

    try:
        # WebViewウィンドウを作成してメインスレッドで実行（URL非表示）
        window = webview.create_window('MixMate', f'http://localhost:{port}', width=1200, height=800)
        api = Api(window)
        webview.start(api, debug=False)  # デフォルトエンジン
    except KeyboardInterrupt:
        pass  # 静かに終了
    except Exception as e:
        print(f"WebView起動エラー: {e}")
        pass

if __name__ == '__main__':
    main()