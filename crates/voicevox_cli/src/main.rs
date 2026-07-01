use std::io::{self, IsTerminal as _, Read, Write as _};
use std::sync::LazyLock;

use anyhow::Context as _;
use camino::Utf8PathBuf;
use clap::Parser;
use flate2::read::GzDecoder;
use futures_util::TryStreamExt as _;
use indicatif::{ProgressBar, ProgressStyle};
use scraper::{Html, Selector};
use tar::Archive;
use voicevox_core::{
    blocking::{Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile},
    AccelerationMode, StyleId,
};

const VOICEVOX_CORE_DIR: &str = "./voicevox_core";

const ONNXRUNTIME_DIR: &str = "./voicevox_core/onnxruntime";
const ONNXRUNTIME_LIB_DIR: &str = "./voicevox_core/onnxruntime/lib";
const ONNXRUNTIME_TERMS_FILE: &str = "TERMS.txt";
const ONNXRUNTIME_TERMS_NAME: &str = "VOICEVOX ONNX Runtime 利用規約";

const ONNXRUNTIME_BUILDER_REPO_OWNER: &str = "VOICEVOX";
const ONNXRUNTIME_BUILDER_REPO_NAME: &str = "onnxruntime-builder";

/// VVM の自動ダウンロード先ディレクトリ
const DEFAULT_VVM_DIR: &str = "./voicevox_core/models/vvms";

/// VOICEVOX/voicevox_vvm GitHub リポジトリ
const MODELS_REPO_OWNER: &str = "VOICEVOX";
const MODELS_REPO_NAME: &str = "voicevox_vvm";
const MODELS_TERMS_FILE: &str = "TERMS.txt";
const MODELS_TERMS_NAME: &str = "VOICEVOX 音声モデル 利用規約";

static DOWNLOAD_PROGRESS_STYLE: LazyLock<ProgressStyle> = LazyLock::new(|| {
    ProgressStyle::with_template(
        "{prefix:40} {bytes:>11} {bytes_per_sec:>13} {elapsed_precise} {bar} {percent:>3}%",
    )
    .unwrap()
});

/// VOICEVOX CLI for text-to-speech synthesis
#[derive(Parser, Debug)]
#[command(name = "voicevox", about = "VOICEVOX CLI for text-to-speech synthesis")]
struct Args {
    /// Text to synthesize. If not specified or "-", read from standard input.
    #[arg(short, long)]
    text: Option<String>,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: Utf8PathBuf,

    /// Path to the voice model file (.vvm).
    /// 省略した場合、VOICEVOX/voicevox_vvm から自動でダウンロードします。
    #[arg(short, long)]
    vvm: Option<Utf8PathBuf>,

    /// Open JTalk dictionary directory path.
    #[arg(short, long)]
    dict: Option<Utf8PathBuf>,

    /// Path to the ONNX Runtime dynamic library file (e.g. libonnxruntime.so.1.19.0).
    #[arg(long)]
    onnxruntime: Option<Utf8PathBuf>,

    /// Style ID for speech synthesis.
    #[arg(short, long, default_value_t = 0)]
    style_id: u32,

    /// Acceleration mode (auto, cpu, gpu).
    #[arg(long, default_value = "auto")]
    acceleration_mode: String,

    /// List all speakers and style IDs in the voice model, then exit.
    /// このオプションを使う場合は --vvm の明示的な指定が必要です。
    #[arg(long)]
    list_styles: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // --list-styles は --vvm の明示指定が必要
    if args.list_styles {
        let vvm_path = args
            .vvm
            .context("--list-styles を使う場合は --vvm で音声モデルのパスを指定してください")?;
        let model = VoiceModelFile::open(&vvm_path)
            .with_context(|| format!("音声モデルの読み込みに失敗しました: {}", vvm_path))?;
        for character in model.metas() {
            println!(
                "Character: {} (UUID: {})",
                character.name, character.speaker_uuid
            );
            for style in &character.styles {
                println!(
                    "  - Style: {} (ID: {}, Type: {:?})",
                    style.name, style.id, style.r#type
                );
            }
        }
        return Ok(());
    }

    // VVM パスを解決（未指定なら自動ダウンロード）
    let vvm_path = resolve_vvm(args.vvm).await?;

    // 音声モデルを開く
    let model = VoiceModelFile::open(&vvm_path)
        .with_context(|| format!("音声モデルの読み込みに失敗しました: {}", vvm_path))?;

    // 合成するテキストを読み込む（未指定または"-"なら標準入力）
    let mut text = args.text;
    if text.is_none() || text.as_deref() == Some("-") {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("標準入力からのテキストの読み込みに失敗しました")?;
        text = Some(buf);
    }
    let text = text.unwrap();

    // Open JTalk 辞書パスを解決
    let dict_dir = match args.dict {
        Some(path) => path,
        None => {
            let paths = [
                Utf8PathBuf::from(format!(
                    "{VOICEVOX_CORE_DIR}/dict/open_jtalk_dic_utf_8-1.11"
                )),
                Utf8PathBuf::from("./crates/test_util/data/open_jtalk_dic_utf_8-1.11"),
            ];
            paths
                .iter()
                .find(|p| p.exists())
                .cloned()
                .context("Open JTalk辞書ディレクトリのパスを指定してください（--dict <PATH>）")?
        }
    };

    // ONNX Runtime ライブラリパスを解決（未指定なら自動ダウンロード）
    let onnxruntime_path = resolve_onnxruntime(args.onnxruntime).await?;

    // ONNX Runtime をロード
    let ort = Onnxruntime::load_once()
        .filename(onnxruntime_path.into_os_string())
        .perform()
        .context("ONNX Runtimeのロードに失敗しました")?;

    // 加速モードを解析
    let accel = match args.acceleration_mode.to_lowercase().as_str() {
        "cpu" => AccelerationMode::Cpu,
        "gpu" => AccelerationMode::Gpu,
        _ => AccelerationMode::Auto,
    };

    // Synthesizer を構築
    let ojt = OpenJtalk::new(dict_dir).context("Open JTalk辞書のロードに失敗しました")?;
    let synth = Synthesizer::builder(ort)
        .text_analyzer(ojt)
        .acceleration_mode(accel)
        .build()
        .context("Synthesizerの構築に失敗しました")?;

    // 音声モデルをロード
    synth
        .load_voice_model(&model)
        .perform()
        .context("音声モデルのロードに失敗しました")?;

    // 音声合成
    let style_id = StyleId(args.style_id);
    let wav = synth
        .tts(&text, style_id)
        .perform()
        .context("音声合成に失敗しました")?;

    // WAV ファイルを書き出す
    std::fs::write(&args.output, wav).context("出力ファイルの書き込みに失敗しました")?;
    eprintln!("Saved generated speech to {}", args.output);

    Ok(())
}

/// ONNX Runtime パスを解決する。
/// - `onnxruntime_arg` が `Some` ならそのまま返す。
/// - `None` なら既知のパスを検索し、見つからなければ自動ダウンロードする。
async fn resolve_onnxruntime(onnxruntime_arg: Option<Utf8PathBuf>) -> anyhow::Result<Utf8PathBuf> {
    if let Some(path) = onnxruntime_arg {
        return Ok(path);
    }

    // 既存の ONNX Runtime ライブラリを探す
    if let Some(existing) = find_onnxruntime()? {
        eprintln!("既存のONNX Runtimeを使用します: {existing}");
        return Ok(existing);
    }

    // なければダウンロード
    download_onnxruntime_with_consent().await
}

/// 既存の ONNX Runtime ライブラリパスを検索する。
fn find_onnxruntime() -> anyhow::Result<Option<Utf8PathBuf>> {
    let paths = [
        Utf8PathBuf::from(format!(
            "{ONNXRUNTIME_LIB_DIR}/{}",
            Onnxruntime::LIB_VERSIONED_FILENAME
        )),
        Utf8PathBuf::from(format!(
            "./target/voicevox_core/downloads/onnxruntime/{}",
            Onnxruntime::LIB_VERSIONED_FILENAME
        )),
    ];
    Ok(paths.iter().find(|p| p.exists()).cloned())
}

/// GitHub から利用規約を表示して同意を得た後、ONNX Runtime をダウンロードして保存する。
async fn download_onnxruntime_with_consent() -> anyhow::Result<Utf8PathBuf> {
    let octocrab = {
        let mut builder = octocrab::Octocrab::builder();
        if let Ok(token) = std::env::var("GH_TOKEN").or_else(|_| std::env::var("GITHUB_TOKEN")) {
            builder = builder.personal_token(token);
        }
        builder.build()?
    };

    // 最新リリースを取得
    eprintln!("GitHub から ONNX Runtime の最新リリース情報を取得しています...");
    let version = Onnxruntime::LIB_VERSION;
    let tag = format!("voicevox_onnxruntime-{version}");
    let release = octocrab
        .repos(ONNXRUNTIME_BUILDER_REPO_OWNER, ONNXRUNTIME_BUILDER_REPO_NAME)
        .releases()
        .get_by_tag(&tag)
        .await
        .with_context(|| format!("`{tag}` リリースの取得に失敗しました"))?;

    // 利用規約を取得（TERMS.txt アセット → リリースボディの順に試行）
    let terms = if let Some(terms_asset) = release.assets.iter().find(|a| a.name == ONNXRUNTIME_TERMS_FILE)
    {
        let terms_bytes = octocrab
            .repos(ONNXRUNTIME_BUILDER_REPO_OWNER, ONNXRUNTIME_BUILDER_REPO_NAME)
            .release_assets()
            .stream(terms_asset.id.0)
            .await
            .context("TERMS.txt のダウンロード開始に失敗しました")?
            .try_fold(Vec::new(), |mut acc, chunk| async move {
                acc.extend_from_slice(&chunk);
                Ok(acc)
            })
            .await
            .context("TERMS.txt のダウンロードに失敗しました")?;
        String::from_utf8(terms_bytes).context("TERMS.txt が UTF-8 ではありません")?
    } else if let Some(body) = &release.body {
        extract_voicevox_onnxruntime_terms(body)
            .context("リリースボディから利用規約を抽出できませんでした")?
    } else {
        anyhow::bail!("利用規約が見つかりませんでした")
    };

    // 利用規約の同意を確認
    ensure_confirmation(ONNXRUNTIME_TERMS_NAME, &terms)?;

    // ダウンロードするアセットを選ぶ
    let asset_name = format!(
        "voicevox_onnxruntime-{}-{version}.tgz",
        onnxruntime_artifact_name(),
    );
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .with_context(|| format!("リリースに `{asset_name}` が見つかりませんでした"))?;

    eprintln!("\nONNX Runtime ({}) をダウンロードしています...", asset.name);

    // 保存先ディレクトリを作成
    fs_err::create_dir_all(ONNXRUNTIME_LIB_DIR)
        .with_context(|| format!("ディレクトリの作成に失敗しました: {ONNXRUNTIME_LIB_DIR}"))?;

    // プログレスバー付きでダウンロード
    let pb = ProgressBar::new(asset.size as u64);
    pb.set_style(DOWNLOAD_PROGRESS_STYLE.clone());
    pb.set_prefix(asset.name.clone());

    let mut bytes_stream = octocrab
        .repos(ONNXRUNTIME_BUILDER_REPO_OWNER, ONNXRUNTIME_BUILDER_REPO_NAME)
        .release_assets()
        .stream(asset.id.0)
        .await
        .context("ONNX Runtime のダウンロード開始に失敗しました")?;

    let mut content: Vec<u8> = Vec::with_capacity(asset.size as usize);
    while let Some(chunk) = bytes_stream.try_next().await? {
        content.extend_from_slice(&chunk);
        pb.set_position(content.len() as u64);
    }
    pb.finish_with_message("完了！");

    // tgz を展開
    eprintln!("ONNX Runtime を展開しています...");
    let lib_path = extract_onnxruntime_tgz(&content)?;

    // TERMS.txt も保存しておく
    let terms_path = Utf8PathBuf::from(ONNXRUNTIME_DIR).join(ONNXRUNTIME_TERMS_FILE);
    let _ = std::fs::write(terms_path.as_std_path(), &terms);

    eprintln!("ONNX Runtime を保存しました: {lib_path}");
    Ok(lib_path)
}

fn onnxruntime_artifact_name() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x64"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "linux-arm64"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "osx-x86_64"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "osx-arm64"
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "win-x64"
    } else if cfg!(all(target_os = "windows", target_arch = "x86")) {
        "win-x86"
    } else {
        panic!("自動ダウンロードに対応していないプラットフォームです")
    }
}

/// リリースボディ（Markdown）の HTML ブロックから利用規約を抽出する。
fn extract_voicevox_onnxruntime_terms(body: &str) -> anyhow::Result<String> {
    let selector = Selector::parse("pre[data-voicevox-onnxruntime-terms] > code")
        .expect("should be valid");
    for node in Html::parse_fragment(body).select(&selector) {
        let text: String = node.text().collect();
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    anyhow::bail!("リリースボディ内に `<pre data-voicevox-onnxruntime-terms><code>` が見つかりませんでした")
}

fn extract_onnxruntime_tgz(tgz: &[u8]) -> anyhow::Result<Utf8PathBuf> {
    let lib_name = Onnxruntime::LIB_VERSIONED_FILENAME;
    let mut lib_content: Option<Vec<u8>> = None;

    let decoder = GzDecoder::new(tgz);
    let mut archive = Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.components().count() >= 2
            && path.components().nth(1).is_some_and(|c| c.as_os_str() == "lib")
            && path
                .file_name()
                .is_some_and(|n| n == lib_name)
        {
            let mut buf = Vec::with_capacity(entry.size() as _);
            entry.read_to_end(&mut buf)?;
            lib_content = Some(buf);
            break;
        }
    }

    let lib_content =
        lib_content.with_context(|| format!("アーカイブ内に `lib/{lib_name}` が見つかりませんでした"))?;

    let lib_path = Utf8PathBuf::from(ONNXRUNTIME_LIB_DIR).join(lib_name);
    fs_err::write(&lib_path, &lib_content)
        .with_context(|| format!("ライブラリの書き込みに失敗しました: {lib_path}"))?;

    Ok(lib_path)
}

/// VVM パスを解決する。
/// - `vvm_arg` が `Some` ならそのまま返す。
/// - `None` なら `DEFAULT_VVM_DIR` 内を検索し、
///   見つからなければ利用規約同意後に自動ダウンロードする。
async fn resolve_vvm(vvm_arg: Option<Utf8PathBuf>) -> anyhow::Result<Utf8PathBuf> {
    if let Some(path) = vvm_arg {
        return Ok(path);
    }

    let vvm_dir = Utf8PathBuf::from(DEFAULT_VVM_DIR);

    // 既存の VVM を探す
    if let Some(existing) = find_vvm_in_dir(&vvm_dir)? {
        eprintln!("既存の音声モデルを使用します: {existing}");
        return Ok(existing);
    }

    // なければダウンロード
    eprintln!("音声モデルが見つかりません。VOICEVOX/voicevox_vvm からダウンロードします。");
    download_vvm_with_consent(&vvm_dir).await
}

/// ディレクトリ内の最初の `.vvm` ファイルを返す。
fn find_vvm_in_dir(dir: &Utf8PathBuf) -> anyhow::Result<Option<Utf8PathBuf>> {
    let read_dir = match std::fs::read_dir(dir.as_std_path()) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    for entry in read_dir {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("vvm") {
            let utf8 = Utf8PathBuf::from_path_buf(path)
                .map_err(|p| anyhow::anyhow!("パスが UTF-8 ではありません: {:?}", p))?;
            return Ok(Some(utf8));
        }
    }
    Ok(None)
}

/// GitHub から利用規約を表示して同意を得た後、VVM をダウンロードして保存する。
async fn download_vvm_with_consent(vvm_dir: &Utf8PathBuf) -> anyhow::Result<Utf8PathBuf> {
    // octocrab クライアントを構築
    let octocrab = {
        let mut builder = octocrab::Octocrab::builder();
        if let Ok(token) = std::env::var("GH_TOKEN").or_else(|_| std::env::var("GITHUB_TOKEN")) {
            builder = builder.personal_token(token);
        }
        builder.build()?
    };

    // 最新リリースを取得
    eprintln!("GitHub から最新リリース情報を取得しています...");
    let release = octocrab
        .repos(MODELS_REPO_OWNER, MODELS_REPO_NAME)
        .releases()
        .get_latest()
        .await
        .context("VOICEVOX/voicevox_vvm の最新リリースの取得に失敗しました\n（ネットワーク接続またはレートリミットを確認してください）")?;

    // TERMS.txt アセットを探す
    let terms_asset = release
        .assets
        .iter()
        .find(|a| a.name == MODELS_TERMS_FILE)
        .context("リリースに TERMS.txt が見つかりませんでした")?;

    // TERMS.txt の内容をダウンロード
    let terms_bytes = octocrab
        .repos(MODELS_REPO_OWNER, MODELS_REPO_NAME)
        .release_assets()
        .stream(terms_asset.id.0)
        .await
        .context("TERMS.txt のダウンロード開始に失敗しました")?
        .try_fold(Vec::new(), |mut acc, chunk| async move {
            acc.extend_from_slice(&chunk);
            Ok(acc)
        })
        .await
        .context("TERMS.txt のダウンロードに失敗しました")?;
    let terms =
        String::from_utf8(terms_bytes).context("TERMS.txt が UTF-8 ではありません")?;

    // 利用規約の同意を確認
    ensure_confirmation(MODELS_TERMS_NAME, &terms)?;

    // ダウンロードする VVM アセットを選ぶ（最初の .vvm ファイル）
    let vvm_asset = release
        .assets
        .iter()
        .find(|a| a.name.ends_with(".vvm"))
        .context("リリースに .vvm ファイルが見つかりませんでした")?;

    eprintln!(
        "\n音声モデル {} をダウンロードしています...",
        vvm_asset.name
    );

    // 保存先ディレクトリを作成
    std::fs::create_dir_all(vvm_dir.as_std_path())
        .with_context(|| format!("ディレクトリの作成に失敗しました: {vvm_dir}"))?;

    // プログレスバー付きでダウンロード
    let pb = ProgressBar::new(vvm_asset.size as u64);
    pb.set_style(DOWNLOAD_PROGRESS_STYLE.clone());
    pb.set_prefix(vvm_asset.name.clone());

    let mut bytes_stream = octocrab
        .repos(MODELS_REPO_OWNER, MODELS_REPO_NAME)
        .release_assets()
        .stream(vvm_asset.id.0)
        .await
        .context("VVM のダウンロード開始に失敗しました")?;

    let mut content: Vec<u8> = Vec::with_capacity(vvm_asset.size as usize);
    while let Some(chunk) = bytes_stream.try_next().await? {
        content.extend_from_slice(&chunk);
        pb.set_position(content.len() as u64);
    }
    pb.finish_with_message("完了！");

    // ファイルに書き込む
    let vvm_path = vvm_dir.join(&vvm_asset.name);
    std::fs::write(vvm_path.as_std_path(), &content)
        .with_context(|| format!("VVM の保存に失敗しました: {vvm_path}"))?;

    // TERMS.txt も保存しておく（再起動時の参照用）
    let terms_path = vvm_dir
        .parent()
        .map(|p| p.join(MODELS_TERMS_FILE))
        .unwrap_or_else(|| Utf8PathBuf::from(MODELS_TERMS_FILE));
    let _ = std::fs::write(terms_path.as_std_path(), &terms);

    eprintln!("音声モデルを保存しました: {vvm_path}");
    Ok(vvm_path)
}

/// 利用規約をページャーで表示し、ユーザーに同意を求める。
fn ensure_confirmation(terms_name: &str, terms: &str) -> anyhow::Result<()> {
    use unicode_width::UnicodeWidthStr as _;

    let max_width = terms.lines().map(|l| l.width()).max().unwrap_or(0);

    let mut terms_pretty = format!(
        "ダウンロードには以下の利用規約への同意が必要です。\n\
         （上下キーとスペースでスクロールし、読み終えたら q を押してください）\n"
    );
    terms_pretty += &format!("─┬─{}\n", "─".repeat(max_width));
    for line in terms.lines() {
        terms_pretty += &format!(" │ {line}\n");
    }
    terms_pretty += &format!("─┴─{}\n", "─".repeat(max_width));

    loop {
        // minus でページャー表示（パニックしたらそのまま print）
        let result = std::panic::catch_unwind(|| {
            minus::page_all({
                let pager = minus::Pager::new();
                pager.set_text(&terms_pretty)?;
                pager.set_prompt(
                    "上下キーとスペースでスクロールし、読み終えたら q を押してください",
                )?;
                pager.set_exit_strategy(minus::ExitStrategy::PagerQuit)?;
                pager
            })
        });

        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("ページャーエラー: {e}");
                print!("{terms_pretty}");
                io::stdout().flush()?;
            }
            Err(_) => {
                eprintln!("ページャーでパニックが発生しました");
                print!("{terms_pretty}");
                io::stdout().flush()?;
            }
        }

        let input = rprompt::prompt_reply_from_bufread(
            &mut io::stdin().lock(),
            &mut io::stderr(),
            format!(
                "[Agreement Required]\n\
                 「{terms_name}」に同意しますか？\n\
                 同意する場合は y を、同意しない場合は n を、再確認する場合は r を入力し、\
                 エンターキーを押してください。\n\
                 [y,n,r] : "
            ),
        )?;

        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => return Ok(()),
            "n" | "no" => {
                anyhow::bail!("利用規約に同意しなかったため、ダウンロードをキャンセルしました")
            }
            "r" => continue,
            _ => {
                if !io::stdin().is_terminal() {
                    anyhow::bail!(
                        "標準入力が TTY ではなく、無効な入力を受け取りました: {:?}",
                        input
                    );
                }
                eprintln!("y, n, r のいずれかを入力してください。");
            }
        }
    }
}
