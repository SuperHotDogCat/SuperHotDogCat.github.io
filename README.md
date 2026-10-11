# SuperHotDogCat's Log Book

[Jekyll](https://jekyllrb.com/) で作ったポートフォリオサイトです。
デザインは [Lei Mao's Log Book](https://leimao.github.io/) を参考にしています。

GitHub Pages は Jekyll に対応しているため、**Markdown ファイルを編集して main ブランチに push するだけでサイトが自動で更新されます。**

## ディレクトリ構成

```
.
├── _config.yml          # サイト全体の設定 (タイトル・プロフィール・URL など)
├── index.md             # ホームページ (Home)
├── blog/index.html      # ブログ一覧ページ (Blog)
├── projects/index.html  # 作ったもの一覧ページ (Projects)
├── pages/               # 追加ページ (Archives, About など) ← ここにページを追加
├── _posts/              # ブログ記事 (Markdown)
│   └── en/              #   英語の記事
├── _projects/           # 作ったもの (Markdown)
│   └── en/              #   英語版
├── en/                  # 英語のページ (日本語版と同じ構成: index.md, blog/, projects/, pages/)
├── _data/i18n.yml       # ナビバーや見出しなど UI の文言 (日本語 / 英語)
├── _layouts/            # ページのレイアウト (HTML テンプレート)
├── _includes/           # ナビバー・サイドバーなどの部品
├── _sass/               # スタイル (色は _sass/_variables.scss で変更)
└── assets/              # CSS・画像
```

## サイトの更新方法

### プロフィールを変更する

`_config.yml` の `author` を編集します。名前・自己紹介・アバター画像・SNS リンクが左サイドバーに表示されます。
アバター画像は `assets/images/` に置いて `avatar` のパスを書き換えてください。

```yaml
author:
  name: "SuperHotDogCat"
  avatar: "/assets/images/avatar.svg"
  bio:
    ja: "自己紹介文"
    en: "Self introduction"
  location: "Tokyo, Japan"
  links:
    - name: GitHub
      icon: "fab fa-github"        # Font Awesome のアイコン名
      url: "https://github.com/SuperHotDogCat"
```

### ホームページを編集する

`index.md` の本文 (`---` で囲まれた部分より下) を Markdown で編集します。
本文の下には「最近の記事」と「作ったもの」が自動で表示されます。

### ブログ記事を追加する

`_posts/` に `YYYY-MM-DD-英数字のスラッグ.md` という名前でファイルを作成します。

```markdown
---
title: "記事のタイトル"
categories: [技術]               # 任意
tags: [Python, 機械学習]          # 任意
description: "一覧に表示される概要"  # 任意 (省略時は本文の冒頭)
math: true                       # 任意: 数式 ($...$, $$...$$) を使う場合
image: /assets/images/xxx.png    # 任意: アイキャッチ画像
---

ここに本文を Markdown で書きます。
```

- URL は `/blog/YYYY/MM/DD/スラッグ/` になります。
- **日付が未来の記事は公開されません。** 下書きは `_drafts/` に置くと公開されず、`bundle exec jekyll serve --drafts` でプレビューだけできます。
- 記事内の画像は `assets/images/` に置き、`![説明](/assets/images/xxx.png)` で参照します。

### 作ったものを追加する

`_projects/` に `スラッグ.md` を作成します。URL は `/projects/スラッグ/` になります。

```markdown
---
title: プロジェクト名
date: 2025-06-01                       # 一覧はこの日付の新しい順
description: 一覧カードに表示される短い説明
image: /assets/images/my-project.png   # 任意: サムネイル (2:1 推奨)
tech: [Python, React]                  # 任意: 使用技術タグ
github: https://github.com/...         # 任意
demo: https://...                      # 任意
featured: true                         # 任意: ホームページに表示
---

詳細ページの本文を Markdown で書きます。
```

### ページを追加する (Home / Blog / Projects 以外)

`pages/` に Markdown (または HTML) ファイルを作成し、front matter に `title`・`permalink`・`nav_order` を書きます。
**`nav_order` を書いたページはナビゲーションバーに自動で追加されます。** 数字の小さい順に左から並びます。

```markdown
---
title: Publications
permalink: /publications/
nav_order: 4
---

# 本文
```

現在の並び順は Home (1), Blog (2), Projects (3), Archives (4), About (5) です。
英語版のナビバーには `en/` 以下のページが表示されるので、英語ページも追加する場合は `en/pages/` に同様に作成します (下記「日本語 / 英語の切り替え」参照)。

| front matter | 説明 |
| --- | --- |
| `title` | ページタイトル (ナビバーにも表示) |
| `permalink` | ページの URL |
| `nav_order` | ナビバーでの順番。書かなければナビバーに表示されない |
| `nav_title` | ナビバーにだけ別の名前を表示したいとき |
| `sidebar: false` | 左右のサイドバーを消して 1 カラム表示にする |
| `hide_title: true` | 本文上部のタイトルを非表示にする |
| `math: true` | MathJax で数式を表示する |

ページを削除するときはファイルを消すだけで、ナビバーからも消えます。

### 日本語 / 英語の切り替え

日本語のページはルート (`/`)、英語のページは `/en/` 以下で公開され、ナビバー右端の **JA / EN** で切り替えられます。
英語版は日本語版と同じ名前・同じ構成のファイルを英語用の場所に置くだけで、自動で対応付けられます。

| 種類 | 日本語 | 英語 |
| --- | --- | --- |
| ホーム | `index.md` | `en/index.md` |
| ブログ記事 | `_posts/2026-10-11-hello-world.md` | `_posts/en/2026-10-11-hello-world.md` (日付・ファイル名を揃える) |
| 作ったもの | `_projects/foo.md` | `_projects/en/foo.md` |
| 追加ページ | `pages/foo.md` (`permalink: /foo/`) | `en/pages/foo.md` (`permalink: /en/foo/`) |

- 切り替えボタンは「URL の先頭に `/en` を付けた (外した) ページ」へリンクします。翻訳がないページでは、その言語のホームへ移動します。
- 一覧・サイドバー・ナビバーには、表示中の言語の記事・ページだけが出ます。片方の言語だけで書いた記事も問題ありません。
- `_posts/en/`・`_projects/en/`・`en/` 以下のファイルは自動で `lang: en` になります (設定は `_config.yml` の `defaults`)。
- プロフィールの自己紹介は `_config.yml` の `author.bio` に言語ごとに書きます。
- 「続きを読む」などの UI の文言は `_data/i18n.yml` で変更できます。

### 見た目を変更する

- 色・フォント: `_sass/_variables.scss`
- レイアウト: `_sass/_layout.scss`、部品: `_sass/_components.scss`
- HTML テンプレート: `_layouts/`, `_includes/`

## ビルド方法 (ローカルでのプレビュー)

Ruby (3.0 以上推奨) と Bundler が必要です。

```bash
gem install bundler
```

依存関係をインストールします (初回のみ)。`vendor/bundle` にインストールしておくとシステムの gem を汚しません。

```bash
bundle config set --local path vendor/bundle
bundle install
```

サイトをビルドします。生成物は `_site/` に出力されます。

```bash
bundle exec jekyll build
```

編集しながらプレビューするには次のコマンドを実行し、<http://127.0.0.1:4000/> にアクセスします。
ファイルを保存すると自動で再ビルドされます (`_config.yml` を変更したときだけは再起動が必要です)。

```bash
bundle exec jekyll serve
# 下書き (_drafts/) も表示する場合
bundle exec jekyll serve --drafts
```

## GitHub Pages への公開

1. GitHub にリポジトリを作成して push します。
   - リポジトリ名を `<ユーザー名>.github.io` にすると `https://<ユーザー名>.github.io/` で公開されます。
   - それ以外の名前 (例: `dogssr`) の場合は `https://<ユーザー名>.github.io/dogssr/` で公開されるため、`_config.yml` の `baseurl` を `"/dogssr"` に変更してください。
2. リポジトリの **Settings → Pages → Build and deployment** で Source を **Deploy from a branch**、Branch を **main / (root)** に設定します。
3. 以降は main ブランチに push するたびに自動でビルド・公開されます (反映まで 1〜2 分)。

`_config.yml` の `url` も公開 URL に合わせて変更してください (RSS フィードや OGP タグで使われます)。
