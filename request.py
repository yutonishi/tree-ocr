import os
import sys
import argparse
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Tuple

# 環境変数を読み込み
load_dotenv()

class Node:
    """ツリー構造のノードを表すクラス"""
    def __init__(self, text: str, level: int, parent: Optional['Node'] = None):
        self.text = text
        self.level = level
        self.parent = parent
        self.children: List[Node] = []
    
    def add_child(self, child: 'Node'):
        """子ノードを追加"""
        self.children.append(child)
        child.parent = self
    
    def to_dict(self) -> Dict[str, Any]:
        """ノード情報を辞書形式で返す"""
        parent_text = self.parent.text if self.parent else None
        return {
            "text": self.text,
            "level": self.level,
            "parent": parent_text,
            "children": [child.to_dict() for child in self.children]
        }
    
    def __str__(self) -> str:
        """ノード情報を文字列として返す"""
        parent_text = self.parent.text if self.parent else "None"
        return f"テキスト: {self.text}, 階層: {self.level}, 親: {parent_text}"

def detect_text(image_path: str) -> List[str]:
    """
    Google Cloud Vision APIを使用して画像からテキストを検出
    """
    try:
        # 環境変数からAPIキーを取得
        api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_CLOUD_API_KEY環境変数が設定されていません。")
        
        # Google Cloud Vision APIに接続するためのサービスアカウントキーファイルを作成
        import json
        import tempfile
        import requests
        from base64 import b64encode
        
        # 画像ファイルを読み込む
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        # 画像をBase64エンコード
        encoded_content = b64encode(content).decode('utf-8')
        
        # Vision APIリクエストを直接作成
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": encoded_content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        # APIリクエストを送信
        response = requests.post(vision_api_url, json=request_data)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} {response.text}")
            
        # レスポンスからテキスト検出結果を取得
        result = response.json()
        
        if not result.get("responses") or not result["responses"][0].get("textAnnotations"):
            print("テキストが検出されませんでした。")
            return []
        
        # OCRの結果からテキスト行を抽出
        # 最初の要素は全テキストなのでスキップ
        text_annotations = result["responses"][0]["textAnnotations"]
        lines = [text["description"] for text in text_annotations[1:]]
        
        return lines
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []

def parse_tree_structure(text_lines: List[str]) -> List[Node]:
    """
    検出されたテキストからツリー構造を解析
    簡易的な実装：インデントの深さでレベルを判断
    """
    if not text_lines:
        return []
    
    # 各行のインデントを計算してレベルを推定
    nodes = []
    root_nodes = []
    last_nodes = [None] * 100  # 各レベルの最後のノードを保持（十分な大きさの配列）
    
    for line in text_lines:
        # 行の先頭のスペースの数を数えてレベルを推定
        stripped_line = line.lstrip()
        indent = len(line) - len(stripped_line)
        level = indent // 2  # 2スペースをインデント1レベルと仮定
        
        # 空行やスペースのみの行はスキップ
        if not stripped_line:
            continue
        
        # ノードを作成
        node = Node(stripped_line, level)
        nodes.append(node)
        
        if level == 0:
            # ルートノード
            root_nodes.append(node)
        elif last_nodes[level-1] is not None:
            # 親ノードに追加
            last_nodes[level-1].add_child(node)
        
        # 現在のレベルの最後のノードを更新
        last_nodes[level] = node
    
    return root_nodes

def print_tree(root_nodes: List[Node], indent: int = 0):
    """再帰的にツリーを表示"""
    for node in root_nodes:
        prefix = "  " * indent
        parent = node.parent.text if node.parent else "None"
        print(f"{prefix}- テキスト: {node.text}")
        print(f"{prefix}  階層: {node.level}")
        print(f"{prefix}  親: {parent}")
        
        if node.children:
            print_tree(node.children, indent + 1)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="画像からツリー構造を抽出するOCRツール")
    parser.add_argument("image_path", help="処理する画像ファイルのパス")
    args = parser.parse_args()
    
    # 画像ファイルが存在するか確認
    if not os.path.exists(args.image_path):
        print(f"エラー: 指定されたファイル '{args.image_path}' が見つかりません。")
        sys.exit(1)
    
    print(f"画像ファイル '{args.image_path}' からテキストを抽出しています...")
    
    # 画像からテキストを検出
    text_lines = detect_text(args.image_path)
    
    if not text_lines:
        print("テキストを抽出できませんでした。別の画像を試してください。")
        sys.exit(1)
    
    print(f"{len(text_lines)}行のテキストが検出されました。")
    
    # テキストからツリー構造を解析
    try:
        root_nodes = parse_tree_structure(text_lines)
        
        if not root_nodes:
            print("ツリー構造を検出できませんでした。")
            sys.exit(1)
        
        print("\n--- 検出されたツリー構造 ---")
        print_tree(root_nodes)
        
    except Exception as e:
        print(f"ツリー構造の解析中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()