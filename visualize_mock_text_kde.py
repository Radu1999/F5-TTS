import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer, list_str_to_idx


def get_mock_sentences():
    """Return small sets of mock sentences for English, Romanian, and Chinese."""
    en = [
        # Original 10
        "The quick brown fox jumps over the lazy dog.",
        "I love exploring new machine learning models.",
        "This project compares multilingual text embeddings.",
        "How is the weather today in the city?",
        "Please schedule a meeting for tomorrow morning.",
        "Music helps me focus while I code.",
        "Large language models can generalize surprisingly well.",
        "The cat slept peacefully on the warm windowsill.",
        "We will evaluate the results using robust metrics.",
        "Reading books expands our understanding of the world.",

        # Added 30+ examples
        "Artificial intelligence is transforming many industries.",
        "The sun sets beautifully over the ocean horizon.",
        "She bought fresh vegetables from the local market.",
        "Can you recommend a good restaurant nearby?",
        "We need to iterate on this design quickly.",
        "The concert last night was absolutely amazing.",
        "Always remember to save your work frequently.",
        "Effective communication is key to success.",
        "He runs five kilometers every single morning.",
        "The aroma of freshly baked bread filled the air.",
        "Learning a new language takes time and effort.",
        "They are discussing the latest financial report.",
        "What time does the train depart from the station?",
        "The children were playing happily in the park.",
        "Don't forget to back up all your important files.",
        "I need a cup of strong coffee right now.",
        "Global warming is a critical issue for our planet.",
        "The scientist published a groundbreaking research paper.",
        "This algorithm is much faster than the previous one.",
        "A journey of a thousand miles begins with a single step.",
        "The museum exhibits historical artifacts from the region.",
        "Could you please pass me the salt and pepper?",
        "Innovation requires both creativity and discipline.",
        "The old castle stood proudly atop the hill.",
        "We should optimize the database queries for performance.",
        "Exercise is essential for a healthy lifestyle.",
        "The meeting will cover budget and timeline adjustments.",
        "It's important to take short breaks throughout the day.",
        "The starry night sky was incredibly clear tonight.",
        "They decided to postpone the event until next week.",
        "Solving complex problems is a rewarding challenge.",
        "Thank you for your valuable input on this matter.",
        "The flowers bloom early in the springtime.",
        "Check the system logs for any potential errors.",
        "I enjoy hiking in the mountains during the summer."
    ]

    ro = [
        # Original 10
        "Vulpea rapidă sare peste câinele leneș.",
        "Îmi place să explorez modele noi de învățare automată.",
        "Acest proiect compară încorporările multilingve ale textului.",
        "Cum este vremea astăzi în oraș?",
        "Te rog programează o întâlnire pentru mâine dimineață.",
        "Muzica mă ajută să mă concentrez când scriu cod.",
        "Modelele mari de limbaj pot generaliza surprinzător de bine.",
        "Pisica a dormit liniștită pe pervazul cald.",
        "Vom evalua rezultatele folosind metrici solide.",
        "Cititul ne extinde înțelegerea asupra lumii.",

        # Added 30+ examples
        "Inteligența artificială transformă multe industrii.",
        "Soarele apune frumos peste orizontul oceanului.",
        "Ea a cumpărat legume proaspete din piața locală.",
        "Îmi poți recomanda un restaurant bun în apropiere?",
        "Trebuie să iterăm rapid asupra acestui design.",
        "Concertul de aseară a fost absolut uimitor.",
        "Nu uitați niciodată să vă salvați lucrul frecvent.",
        "Comunicarea eficientă este cheia succesului.",
        "El aleargă cinci kilometri în fiecare dimineață.",
        "Aroma de pâine proaspăt coaptă a umplut aerul.",
        "Învățarea unei limbi noi necesită timp și efort.",
        "Ei discută cel mai recent raport financiar.",
        "La ce oră pleacă trenul din gară?",
        "Copiii se jucau fericiți în parc.",
        "Nu uitați să faceți backup la toate fișierele importante.",
        "Am nevoie de o ceașcă de cafea tare chiar acum.",
        "Încălzirea globală este o problemă critică pentru planeta noastră.",
        "Omul de știință a publicat o lucrare de cercetare revoluționară.",
        "Acest algoritm este mult mai rapid decât cel precedent.",
        "O călătorie de o mie de mile începe cu un singur pas.",
        "Muzeul expune artefacte istorice din regiune.",
        "Puteți să-mi dați, vă rog, sarea și piperul?",
        "Inovația necesită atât creativitate, cât și disciplină.",
        "Vechiul castel stătea mândru în vârful dealului.",
        "Ar trebui să optimizăm interogările bazei de date pentru performanță.",
        "Exercițiul este esențial pentru un stil de viață sănătos.",
        "Întâlnirea va acoperi ajustările de buget și de calendar.",
        "Este important să luați scurte pauze pe parcursul zilei.",
        "Cerul înstelat a fost incredibil de senin în seara asta.",
        "Ei au decis să amâne evenimentul până săptămâna viitoare.",
        "Rezolvarea problemelor complexe este o provocare satisfăcătoare.",
        "Vă mulțumesc pentru contribuția dumneavoastră valoroasă în această privință.",
        "Florile înfloresc devreme primăvara.",
        "Verificați jurnalele de sistem pentru orice erori potențiale.",
        "Îmi place să fac drumeții la munte vara."
    ]

    zh = [
        # Original 10
        "敏捷的棕色狐狸跳过了那只懒狗。",
        "我喜欢探索新的机器学习模型。",
        "这个项目比较多语言文本嵌入。",
        "今天城市里的天气怎么样？",
        "请安排明天早上的会议。",
        "听音乐能让我在编程时更专注。",
        "大型语言模型具有令人惊讶的泛化能力。",
        "那只猫在温暖的窗台上安静地睡着了。",
        "我们将使用可靠的指标来评估结果。",
        "读书能拓展我们对世界的理解。",

        # Added 30+ examples
        "人工智能正在改变许多行业。",
        "夕阳在海平面上落得很美。",
        "她从当地市场买了新鲜蔬菜。",
        "你能推荐附近一家好餐馆吗？",
        "我们需要快速迭代这个设计。",
        "昨晚的音乐会真是太棒了。",
        "务必经常保存你的工作。",
        "有效的沟通是成功的关键。",
        "他每天早上跑五公里。",
        "新鲜出炉的面包的香气弥漫在空气中。",
        "学习一门新语言需要时间和努力。",
        "他们正在讨论最新的财务报告。",
        "火车什么时候从车站出发？",
        "孩子们在公园里玩得很开心。",
        "别忘了备份所有重要文件。",
        "我现在需要一杯浓咖啡。",
        "全球变暖是我们星球面临的关键问题。",
        "这位科学家发表了一篇具有突破性的研究论文。",
        "这个算法比以前的快得多。",
        "千里之行，始于足下。",
        "博物馆展出了该地区的历史文物。",
        "请把盐和胡椒递给我好吗？",
        "创新需要创造力，也需要纪律。",
        "这座古老的城堡傲然矗立在山顶上。",
        "我们应该优化数据库查询以提高性能。",
        "锻炼对健康的生活方式至关重要。",
        "会议将涉及预算和时间表的调整。",
        "在一天中抽出时间休息是很重要的。",
        "今晚的星空异常清晰。",
        "他们决定将活动推迟到下周。",
        "解决复杂的问题是一个有益的挑战。",
        "感谢您对此事的宝贵意见。",
        "春天的时候花开得早。",
        "检查系统日志以查找任何潜在错误。",
        "我喜欢在夏天去爬山徒步。"
    ]

    return {"en": en, "ro": ro, "ro_no_lang": ro, "zh": zh}


def load_cfm_model(device: str = "cpu"):
    """Load the trained Romanian F5-TTS model and return embedder function"""
    print("Loading Romanian F5-TTS model...")

    # Model configuration for F5TTS_v1_Base (from setting.json)
    model_cfg = {
        'dim': 1024,
        'depth': 22,
        'heads': 16,
        'ff_mult': 2,
        'text_dim': 512,
        'conv_layers': 4
    }

    # Paths
    ckpt_path = r"checkpoints\swara_te_doro_lang_novq_40500.pt"

    # Load tokenizer
    vocab_char_map, vocab_size = get_tokenizer("ro_tts", "pinyin")

    print(f"Vocab size: {vocab_size}")

    # Create model
    model = CFM(
        transformer=DiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=100
        ),
        vocab_char_map=vocab_char_map
    )

    # Load model weights
    if ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device="cpu")

        # Handle different checkpoint formats
        if 'ema_model_state_dict' in checkpoint:
            state_dict = checkpoint['ema_model_state_dict']
        else:
            state_dict = checkpoint

        # Remove ema_model prefix if present
        state_dict = {k.replace("ema_model.", ""): v for k, v in state_dict.items()
                      if not k.endswith(('initted', 'update', 'step'))}
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get('ema_model_state_dict', checkpoint)

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    print(f"Model loaded successfully on {device}")

    def encode_texts(texts, language="en"):
        """Encode texts using the CFM model's embedders
        
        For Romanian with language module (ro): use language_module (context-aware)
        For Romanian without language module (ro_no_lang): use text_embed directly (character-level)
        For other languages: use text_embed.text_embed (character-level)
        """
        embeddings_list = []
        
        with torch.no_grad():
            for text in texts:
                # Convert text to indices using the vocab_char_map
                text_indices = list_str_to_idx([text], vocab_char_map)[0]
                text_tensor = torch.tensor(text_indices, device=device).unsqueeze(0)
                
                if language == "ro":
                    # For Romanian with language module: use language_module (contextual embeddings)
                    ro_lang_emb, _, _ = model.transformer.language_module(text_tensor, seq_len=len(text))
                    # Average pool over sequence length to get sentence embedding
                    ro_lang_emb = model.transformer.text_embed(text_tensor, text_embed=ro_lang_emb, seq_len=text_tensor.shape[1])
                    sentence_embedding = ro_lang_emb.mean(dim=1)  # [1, text_dim]
                elif language == "ro_no_lang":
                    # For Romanian without language module: use text_embed directly (character embeddings)
                    # Get embeddings for each character index
                    char_embeddings = model.transformer.text_embed(text_tensor, seq_len=text_tensor.shape[1])
                    # Average pool over sequence length to get sentence embedding
                    sentence_embedding = char_embeddings.mean(dim=1)  # [1, embed_dim]
                else:
                    # For other languages: use text_embed.text_embed (character embeddings)
                    # Get embeddings for each character index
                    char_embeddings = model.transformer.text_embed(text_tensor, seq_len=text_tensor.shape[1])
                    # Average pool over sequence length to get sentence embedding
                    sentence_embedding = char_embeddings.mean(dim=1)  # [1, embed_dim]
                
                embeddings_list.append(sentence_embedding.cpu().numpy()[0])
        
        return np.array(embeddings_list)
    
    return encode_texts


def reduce_to_2d(features: np.ndarray) -> np.ndarray:
    """Reduce features to 2D using PCA via SVD (no sklearn dependency)."""
    # Center
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    # SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    projected = centered @ components
    return projected


def plot_kde_contours(coords_by_lang: dict, output_dir: str):
    """Create joint KDE plot with contours and marginals, similar to the example."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting data for languages: {list(coords_by_lang.keys())}")
    for lang, coords in coords_by_lang.items():
        print(f"  {lang}: {coords.shape}, range X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

    # Build a combined figure with joint distribution
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Colors for languages - more distinct colors
    color_map = {"en": "#e74c3c", "ro": "#3498db", "ro_no_lang": "#f39c12", "zh": "#2ecc71"}  # Red, Blue, Orange, Green
    labels = {"en": "English", "ro": "Romanian (w/ Lang Module)", "ro_no_lang": "Romanian (w/o Lang Module)", "zh": "Chinese"}

    # Add scatter points first, then KDE contours
    for lang, coords in coords_by_lang.items():
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Scatter plot with some transparency
        ax.scatter(x, y, color=color_map[lang], alpha=0.6, s=50, label=f"{labels[lang]} (points)", zorder=3)
        
        # KDE contours - only if we have enough points
        if len(x) >= 3:
            try:
                sns.kdeplot(x=x, y=y, levels=5, color=color_map[lang], fill=False, 
                           linewidths=1.5, ax=ax, alpha=0.8, zorder=2)
            except:
                print(f"  Warning: Could not create KDE for {lang}")

    ax.set_xlabel("Projection on Direction 1", fontsize=12)
    ax.set_ylabel("Projection on Direction 2", fontsize=12)
    ax.legend(title="Language", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.title("Text Embeddings: Language and Architecture Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kde_contours.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "kde_contours.pdf"), bbox_inches='tight')
    plt.show()

    # Jointplot-style layout with marginals
    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(4, 4, wspace=0.1, hspace=0.1)
    ax_main = fig.add_subplot(grid[1:, :3])
    ax_top = fig.add_subplot(grid[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(grid[1:, 3], sharey=ax_main)

    for lang, coords in coords_by_lang.items():
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Main plot: scatter + KDE
        ax_main.scatter(x, y, color=color_map[lang], alpha=0.6, s=30, label=labels[lang])
        if len(x) >= 3:
            try:
                sns.kdeplot(x=x, y=y, levels=4, color=color_map[lang], fill=False, 
                           linewidths=1.2, ax=ax_main, alpha=0.7)
            except:
                pass
        
        # Marginal distributions
        sns.histplot(x=x, color=color_map[lang], ax=ax_top, alpha=0.5, kde=True, stat="density")
        sns.histplot(y=y, color=color_map[lang], ax=ax_right, alpha=0.5, kde=True, stat="density", orientation="horizontal")

    ax_main.set_xlabel("Projection on Direction 1", fontsize=12)
    ax_main.set_ylabel("Projection on Direction 2", fontsize=12)
    ax_main.legend(title="Language", fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_top.set_ylabel("Density", fontsize=10)
    ax_right.set_xlabel("Density", fontsize=10)
    
    plt.suptitle("Text Embeddings: Joint Distribution with Marginals (Lang vs No-Lang Module)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "kde_joint.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "kde_joint.pdf"), bbox_inches='tight')
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sentences = get_mock_sentences()
    print(f"Generated {sum(len(sents) for sents in sentences.values())} total sentences across {len(sentences)} languages")

    encode = load_cfm_model(device)

    coords_by_lang = {}
    for lang, sents in sentences.items():
        print(f"Processing {lang} sentences...")
        embeddings = encode(sents, language=lang)
        print(f"  Embeddings shape: {embeddings.shape}")
        coords = reduce_to_2d(embeddings)
        print(f"  2D coordinates shape: {coords.shape}")
        coords_by_lang[lang] = coords

    print("Creating KDE visualizations...")
    plot_kde_contours(coords_by_lang, output_dir="text_embedding_visualizations")
    print("Done! Check text_embedding_visualizations/ for output files.")


if __name__ == "__main__":
    main()


