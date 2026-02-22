# Learning from Random LoRA-Finetuning on a Song Synthesis Diffusion Model

After trying out ACE-Step v1.5 last weekend, an open-source diffusion model that generates full songs from text prompts: 
https://github.com/ace-step/ACE-Step-1.5, I noticed something that kept bugging me. The model produces genuinely impressive music: the melodies work, the instruments sound right, and the structure makes sense. But the lyrics? sometimes garbled, mumbled, or sometimes just completely wrong. You give it a verse, and what comes out sounds like singing, except none of the words were sometimes the words you asked for.

So I just picked a random singing voice dataset from HuggingFace on a random friday night and ran some even more random LoRA experiments!

Here's an example of what that sounds like - a baseline generation where the lyrics come out completely garbled:

[baseline_garbled_lyrics.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/eval_output_5way/baseline/bf556309-0c7c-4cc9-3af8-658952c0e476.flac)

Quick fun disclaimer before we dive in: in this post, **ASR** means **Attention Sharpness Regularization**, not **Automatic Speech Recognition**.

**This post is about that experiment. What I tried, what happened, what broke, and what I accidentally learned about how a 2-billion-parameter diffusion transformer processes text. None of this is production-ready. The model got worse at most things while getting better at few others. But I learned a lot by poking at it.**

---

## The Setup: What is ACE-Step and Why Are Its Lyrics A Bit Bad (Just my intuition)?

ACE-Step v1.5 is a 2-billion parameter music generation model built on a Diffusion Transformer (DiT) architecture. You give it a text caption describing the style ("acoustic folk song with warm vocals and gentle guitar") and lyrics marked up with structural tags (`[Verse]`, `[Chorus]`), and it generates a complete song with vocals.

To understand the experiments in this post, we need to understand how the model actually works - what the lyrics go through before they influence the generated audio, and where the audio generation happens. This matters because every experiment we ran touches specific parts of this pipeline.

### The Architecture: How a Song Gets Made

ACE-Step generates a song in three phases, each handled by a different model:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ACE-Step v1.5 Pipeline                             │
│                                                                      │
│  INPUTS: Caption ("folk song, warm vocals") + Lyrics ("[Verse]...")   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ PHASE 1: Language Model (Qwen3-0.6B, 1.7B params)             │  │
│  │ Reads the prompt → generates metadata (BPM, key, duration)    │  │
│  │ → generates 5Hz semantic audio codes (high-level plan)        │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                              │                                       │
│  ┌────────────────────────── │ ────────────────────────────────────┐  │
│  │ PHASE 2: DiT Decoder (24-layer Diffusion Transformer, 2B)     │  │
│  │                          ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │ Condition Encoder (packs all text inputs for the DiT)   │   │  │
│  │  │                                                         │   │  │
│  │  │  Caption ──→ Qwen3 full model ──→ linear ──→ ~256 tokens│   │  │
│  │  │  Lyrics  ──→ Qwen3 embed_tokens ──→ 8-layer ──→ ~512   │   │  │
│  │  │              (embedding only!)      bidir.     tokens   │   │  │
│  │  │  Timbre  ──→ TimbreEncoder ──────────────────→ 1 token  │   │  │
│  │  │                                                         │   │  │
│  │  │  All packed into: encoder_hidden_states [B, 769, 2048]  │   │  │
│  │  └──────────────────────────┬──────────────────────────────┘   │  │
│  │                              │                                 │  │
│  │  Noise ──→ [24 DiT layers] ──→ Denoised audio latents         │  │
│  │             Each layer:                                        │  │
│  │               Self-Attention (audio ↔ audio)                   │  │
│  │               Cross-Attention (audio → packed conditions)      │  │
│  │               MLP                                              │  │
│  │             × 8 denoising steps                                │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                              │                                       │
│  ┌────────────────────────── │ ────────────────────────────────────┐  │
│  │ PHASE 3: VAE Decoder                                           │  │
│  │ Audio latents [B, T, 64] → 48kHz stereo waveform               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### The Lyrics Path: Where Words Become Sound

Here's the specific path lyrics take through the model. This is important because our experiments directly affect how well this path works.

**Step 1: Tokenization.** The lyrics string (`"[verse]\nwalking down the morning road..."`) gets tokenized by Qwen3's tokenizer into up to 512 token IDs.

**Step 2: Shallow embedding.** Those token IDs pass through `embed_tokens` - just the embedding lookup table of Qwen3-0.6B. This converts each token ID into a 1024-dimensional vector. Critically, this is *not* the full Qwen3 transformer - no self-attention, no contextual understanding. Each word gets embedded independently, as if the other words don't exist.

**Step 3: LyricEncoder.** The raw embeddings go through an 8-layer bidirectional transformer with rotary position embeddings. This is where the lyrics first gain contextual relationships - "walking" knows it's followed by "down" and preceded by "[verse]". The output is projected to 2048 dimensions: `[B, 512, 2048]`.

**Step 4: Packing.** The lyric embeddings (512 tokens), timbre embedding (1 token), and caption embeddings (~256 tokens) get concatenated and sorted (real tokens first, padding last) into a single packed sequence: `encoder_hidden_states [B, 769, 2048]`.

**Step 5: Cross-attention.** At each of the 24 DiT layers, the audio latents being denoised (the "queries") attend to this packed conditioning sequence (the "keys" and "values"). This is the *only* mechanism through which lyrics influence the generated audio. At each time position in the audio, the model computes attention weights over all 769 conditioning tokens and uses those weights to pull in the relevant information.

The cross-attention computes (simplified):

```
Q = W_q · audio_latent       [B, T, 16 heads, 128 dim]
K = W_k · packed_conditions   [B, 769, 8 heads, 128 dim]
V = W_v · packed_conditions   [B, 769, 8 heads, 128 dim]

attention_weights = softmax(Q · K^T / √128)   [B, heads, T, 769]
output = attention_weights · V
```

This is simplified. In practice, ACE-Step uses grouped-query attention (GQA) with 8 KV heads shared across 16 query heads, and the 24 self-attention layers alternate between sliding window (window=128, odd layers) and full bidirectional attention (even layers). Cross-attention is always full (no sliding window). But the core idea is the same: audio queries attend over conditioning tokens to pull in text information.

This is where the problem lives. Those attention weights have to figure out, at every moment in the audio, which of the 769 tokens to focus on. For the lyrics to come through clearly, the attention at time t (say, 2 seconds into the song) needs to focus on the specific lyric tokens being sung at that moment. Instead, the attention tends to be diffuse - spread thinly across many tokens - and the model produces something that sounds like singing but with garbled words.

### Why Is This So Hard? (Just Hypothesis Again)

Compare the two text inputs - caption vs. lyrics - and notice the asymmetry in how they're encoded:

| Input | Encoder | Depth | What it captures |
|-------|---------|-------|-----------------|
| Caption ("clean female alto vocals...") | Full Qwen3-0.6B (28-layer transformer) | Deep contextual understanding | Global style, mood, instrumentation |
| Lyrics ("[verse]\nwalking down...") | `embed_tokens` → 8-layer bidirectional transformer | Shallow | Word identities and local context |

The caption gets the full power of a pre-trained language model. The lyrics get a much shallower encoding. This is by design (encoding 512 lyric tokens through the full 28-layer Qwen3 at every denoising step would be expensive), but it means the lyric representations entering cross-attention are less rich than the caption representations. The DiT decoder has to do more work to extract phonetic information from weaker representations.

---

## Experiment 1: LoRA Fine-Tuning on Clean Singing Voice Data

### The Idea

I didn't go into this with a sophisticated plan. The thinking was simple: the model produces garbled vocals. I have a dataset of very clean, professional singing voice recordings. What if I just LoRA-fine-tune the model on these recordings paired with descriptive prompts like "clean female alto vocals, studio recording" - would the model learn to produce cleaner vocal output in general?

That's it. No clever alignment strategy. No carefully engineered lyrics. Just: here's what clean singing sounds like, learn from it.

### The Data

I used the English subset of GTSinger - a professional singing voice corpus with three singers (two female altos, one male tenor), five singing techniques (breathy, vibrato, glissando, mixed voice, pharyngeal, plus neutral control), and word-level phonetic annotations for every recording. All clean, studio-quality, 48kHz.

After preprocessing - filtering, deduplication, and concatenating short clips - I ended up with about 4,315 samples totaling roughly 11.7 hours of audio.

Here's the important part that I want to be upfront about: **the lyrics field for every single training sample was just `[verse]` - the structural tag alone, with no actual words.** The preprocessing pipeline tried to extract word sequences from GTSinger's JSON sidecar files, but the key it was looking for didn't match the actual data format. So every sample went through with an empty lyrics field, wrapped in a `[verse]` tag.

What the model *did* get for each sample was a rich, descriptive caption:

> `"clean studio acapella recording, female alto vocals, breathy singing technique, neutral emotion, moderate pace, normal range, English lyrics, professional singer, high quality 48kHz recording"`

And a tag-style genre prompt:

> `"acapella, clean vocals, female vocals, alto, breathy, English, studio recording, professional singer, neutral"`

So in practice, the training data looked like this:

| Input | Value |
|-------|-------|
| Audio | Professional studio singing voice (48kHz, clean) |
| Caption | Rich vocal description ("clean female alto vocals, breathy technique...") |
| Lyrics | `[verse]` (no actual words) |

The model was learning to associate descriptive vocal prompts with clean singing audio - but without any word-level lyric correspondence.

I only realized this after running this experiment and going back to check the dataset - I had just given `[verse]` as the lyrics for every sample. That's what makes this a genuinely random experiment. 😅

### How LoRA Works: Adapting a 2B Model with 5.5M Parameters

Before getting into what happened, it's worth understanding *what* LoRA actually does to the architecture described above.

In each of those 24 DiT decoder layers, the self-attention and cross-attention both use four large weight matrices: W_q, W_k, W_v, and W_o (query, key, value, and output projections). These are `[2048, 2048]` matrices - 4 million parameters each. Normally, fine-tuning would update all of them directly. LoRA instead freezes the original weights and adds a small "side path":

```
Original:  output = W · x                    (2048 × 2048 = 4M params)
With LoRA: output = W · x + (B · A) · x      (A: 2048×8, B: 8×2048 = 33K params)
```

Matrix A projects the 2048-dim input down to rank 8, and matrix B projects it back up. The product B·A is a rank-8 update to the original weight matrix. The key insight: you're not replacing the pre-trained knowledge in W - you're adding a low-rank correction on top of it.

Referring back to the architecture diagram, here's exactly where LoRA adapters sit:

```
Each of 24 DiT decoder layers:
  Self-Attention:
    W_q · audio_latent + (B_q · A_q) · audio_latent    ← LoRA on Q
    W_k · audio_latent + (B_k · A_k) · audio_latent    ← LoRA on K
    W_v · audio_latent + (B_v · A_v) · audio_latent    ← LoRA on V
    W_o · attn_out     + (B_o · A_o) · attn_out        ← LoRA on O
    
  Cross-Attention:
    W_q · audio_latent + (B_q · A_q) · audio_latent    ← LoRA on Q
    W_k · packed_conds + (B_k · A_k) · packed_conds    ← LoRA on K
    W_v · packed_conds + (B_v · A_v) · packed_conds    ← LoRA on V
    W_o · attn_out     + (B_o · A_o) · attn_out        ← LoRA on O
    
  MLP: FROZEN (no LoRA)
```

That's 8 LoRA pairs per layer × 24 layers = **192 adapted weight matrices**. Total trainable parameters: ~5.5M out of 2B (~0.3% of the model). The adapter file is just 22MB.

Nothing else was touched - the LM planner (Phase 1), the VAE decoder (Phase 3), the Qwen3 text encoder, the LyricEncoder, and all MLP layers in the DiT were kept frozen. The *only* thing being updated are the attention projections inside the DiT decoder.

Why does this matter? Look at the cross-attention LoRA adapters specifically:
- **LoRA on K and V** modifies *how the packed conditioning tokens are represented* as keys and values. This changes what information the audio can extract from the lyrics/caption/timbre.
- **LoRA on Q** modifies *how the audio latents formulate their queries*. This changes what the audio "asks for" when it attends to the conditioning.
- **LoRA on O** modifies *how the extracted information gets projected back* into the audio representation.

So LoRA is adjusting both sides of the cross-attention conversation: what the audio asks for, and what the conditioning provides. It's a small tweak to the communication channel between "what the model knows about the text" and "what the model is generating as audio."

### Training

- **Loss:** Standard flow matching (predict the velocity field v = noise - clean_latent)
- **Timestep sampling:** Uniformly from the 8-step turbo schedule {1.0, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300} - this is critical because the model at inference time only uses these 8 discrete timesteps, not a continuous range
- **Optimizer:** AdamW, lr=1e-4, cosine schedule with 100-step warmup
- **Batch:** 1 sample with gradient accumulation over 4 steps (effective batch size 4)
- **Epochs:** 100 over the full dataset
- **Hardware:** Single NVIDIA A10G (24GB), ~4 hours total

The loss converged quickly - dropping from ~0.8 to ~0.45 in the first 500 steps, then stabilizing.

### Quick Note on Module Selection

I tested targeting different combinations of modules during development. Cross-attention alone produced fragmented vocal rendering - the voice would start and stop unnaturally. Self-attention alone helped temporal coherence but didn't improve vocal quality. The combination works best. Adding MLP layers on top (tripling the parameter count to 18M) didn't help - probably overfitting on the dataset.

---

## Evaluating the LoRA Model: What Actually Happened

### The Evaluation Protocol

I built an automated evaluation pipeline across 30 test prompts, each with a unique musical genre (folk, pop, rock, jazz, R&B, country, EDM, choral, indie, reggae, blues, synthwave, gospel, punk, bossa nova, hip-hop, lo-fi, power ballad, Latin pop, singer-songwriter, funk, lullaby, alt-rock, soul, Celtic, tropical house, grunge, disco, ambient, new wave). Each prompt had a genre-appropriate caption, tags, and four lines of original lyrics.

Note the irony: the evaluation prompts had full lyrics, but the training data didn't. At inference time, the model receives real lyrics in the conditioning - it just never saw lyrics-paired-with-audio during fine-tuning.

For each prompt, I generated songs using the baseline model and the LoRA model with the same random seed, same parameters, same everything - the only difference was whether the LoRA adapter was loaded.

I measured two things:

1. **Word Error Rate (WER) and Character Error Rate (CER):** I ran the generated audio through Whisper large-v3 (a speech recognition model) and compared the transcription against the ground truth lyrics. Lower is better. WER=0% means Whisper heard the exact right words. WER=100% means nothing matched.

2. **Attention Alignment Score (AAS):** A metric built into ACE-Step that extracts cross-attention maps from specific DiT layers and measures how well the attention follows a monotonic, diagonal pattern using Dynamic Time Warping. Higher is better.

### The Numbers

But just to be really upfront here, these numbers were on a really, really small sample size, sometimes just 1 sample for some genre, so they are in no way representative of the model’s capabilities. This is just a fun experiment to try out the model and see what sticks.

| Model | WER (%) | CER (%) | AAS (DiT) |
|-------|---------|---------|-----------|
| Baseline (no LoRA) | 61.03 | 51.91 | 0.1182 |
| **+ LoRA** | **46.51** | **39.08** | **0.1498** |

That's a **23.8% relative WER reduction** and a **24.7% relative CER reduction**. The attention alignment also improved by about 27%.

Wait - how? The model was never shown actual lyrics during training. How did lyric intelligibility improve?

I think what happened is this: the LoRA adapter learned to produce cleaner, more articulate vocal rendering from the studio recordings. When the model then receives real lyrics at inference time (through the conditioning that was always there in the base model), the improved vocal clarity makes those lyrics more intelligible - even though the adapter itself has no concept of lyric alignment. It's like giving someone a better microphone doesn't change what they're saying, but it makes the words easier to understand.

The attention alignment score improving is more puzzling. My best guess is that clearer vocal rendering produces latent representations that are more compatible with the attention patterns the base model already learned, creating a feedback loop where better audio quality leads to slightly better attention behavior.

To be clear: these numbers are still not great in absolute terms. A WER of 46.5% means roughly half the words are wrong. But the fact that any improvement happened at all with no lyric data in training is the interesting part.

### Where It Helped Most

The biggest wins came in genres where the baseline either failed completely or produced heavily corrupted lyrics:

| Genre | Baseline WER | LoRA WER | What Changed |
|-------|-------------|----------|--------------|
| Grunge | 1.00 | 0.10 | Went from silence to nearly complete lyrics with minor word-level substitutions |
| Funk | 1.00 | 0.13 | Went from "Thanks for watching!" to near-complete correct lyrics |
| Folk | 1.00 | 0.15 | Went from "Thank you for watching!" to a mostly correct verse |
| Pop | 0.85 | 0.00 | Went from garbled lines to exact ground-truth lyrics |
| Rock | 0.91 | 0.17 | Went from short fragments to mostly correct 4-line output |
| Alt-rock | 1.00 | 0.26 | Went from instrumental-only output to recognizable sung lyrics |
| Soul | 1.00 | 0.28 | Went from silence to mostly correct lyrical structure |
| Power ballad | 0.70 | 0.09 | Went from heavily distorted phrasing to near-complete lines |

My interpretation: the LoRA adapter is doing two useful things at once in these cases. In some genres, it flips the model from degenerate/non-vocal behavior into actually singing. In others, where the model already attempts vocals, it improves articulation enough for Whisper to recover many more words.

### Where It Hurt

Here's the uncomfortable part. In several genres, the LoRA adapter made things worse:

| Genre | Baseline WER | LoRA WER | What Happened |
|-------|-------------|----------|---------------|
| Singer-songwriter | 0.11 | 2.32 | Went from mostly correct lyrics to a long hallucinated word-salad |
| R&B | 0.14 | 1.00 | Went from mostly correct lines to "So So So So So" |
| Lullaby | 0.59 | 1.00 | Went from partially correct lyrics to silence |
| Hip-hop | 0.73 | 1.00 | Went from partial lyrics to "adult rhymes" |
| EDM | 0.00 | 0.27 | Went from perfect transcription to noticeable substitutions |
| Indie | 0.55 | 0.80 | Went from partial structure to short unrelated fragments |

This also makes sense in context. The LoRA adapter was trained exclusively on clean acapella studio vocals - solo singers with no instruments, no effects, no production. It biases the model toward that specific kind of output. For genres where the base model already had stable, style-specific phrasing, that bias can interfere and create regressions.

The model got better at some things by getting worse at others. This is exactly the kind of trade-off you'd expect from fine-tuning on a narrow distribution without any mechanism to preserve the original capabilities. I was okay with this going in - the point was to learn, not to ship.

### Some Entertaining Transcription Examples

The Whisper transcriptions are sometimes accidentally poetic:

- **Baseline Folk:** Whisper heard "Thank you for watching!" instead of the verse
- **Baseline Funk:** Whisper heard "Thanks for watching!" instead of lyrics
- **LoRA R&B:** Collapsed to "So So So So So"
- **LoRA Singer-songwriter:** Produced a very long multilingual hallucination with almost no overlap with the target lyrics

The Pop result is a clean comparison. Ground truth was "I can feel the rhythm tonight / Dancing underneath the stars / Every heartbeat burning bright / Nothing in the world is ours." The baseline produced a garbled version ("I can feel the rhythm to die ..."), while the LoRA version matched all four lines exactly (WER 0.00).

---

## Looking Inside: Cross-Attention Heatmaps

To understand what the LoRA adapter is actually doing inside the model, I extracted cross-attention maps from the DiT decoder during inference and plotted them as heatmaps.

### How to Read These Plots

- **Y-axis (vertical):** The lyric tokens, from top to bottom in order. You can see the actual words on the left side - "[Verse]", "\n", "Walking", "down", "the", "morning", "road", etc. (These are the lyrics from the *evaluation* prompt - not the training data.)
- **X-axis (horizontal):** Time in audio frames at 25 Hz. Frame 0 is the start of the song, frame ~250 is the end of a 10-second clip.
- **Color:** How much attention the audio at time position X pays to lyric token Y. Bright yellow = high attention, dark purple = near zero.
- **What you'd ideally want:** A diagonal pattern from top-left to bottom-right, meaning "at the start of the song, attend to the first lyrics; in the middle, attend to the middle lyrics; at the end, attend to the last lyrics."

Each plot has five panels showing all model variants side by side: Baseline, LoRA (no lyrics), LoRA+ASR (no lyrics), LoRA with lyrics, and LoRA with lyrics+ASR. The first three panels relate to the Part 1 experiments discussed in this section. The last two show the Part 2 experiments (training with actual lyrics) - we analyze those panels in detail further below.

### Pop

![Cross-Attention Maps - Pop](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_pop.png)

A useful example. The **Baseline** (left) shows a visible diagonal pattern - you can see attention shifting from "I" / "can" / "feel" in the early frames to "Nothing" / "in" / "the" / "world" in the later frames. There's also a bright horizontal band (an "attractor" token - usually a newline or structural tag - that gets attention at all time positions).

The **LoRA** (center) is qualitatively similar but the distribution has shifted. The diagonal is still there but different in detail.

Baseline pop WER was 0.85, while LoRA reached 0.00 (perfect transcription on this evaluation). Despite similar-looking averaged maps, the output quality is clearly different.

### Folk: Rescuing a Failed Generation

![Cross-Attention Maps - Folk](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_folk.png)

Baseline folk WER was 1.00 - Whisper heard "Thank you for watching!" instead of the lyrics. The LoRA version achieved 0.15 and produced a mostly correct verse with minor substitutions.

The attention maps show why this is interesting. The **Baseline** has attention spread across multiple lyric tokens but still generated an unrelated phrase. The **LoRA** version has a different pattern and produced the intended verse. The LoRA adapter changed the model's behavior regarding *what lyric content gets rendered*, not just *how well to align lyrics*. That's consistent with what the adapter actually learned: "when there's a clean vocal description, produce the described vocals."

### R&B

![Cross-Attention Maps - R&B](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_rnb.png)

The **Baseline** (left) shows a structured pattern and produced mostly correct lyrics (WER 0.14), with minor substitutions like "Baby, sign me" instead of "Lay beside me."

The **LoRA** (center) has a different pattern but collapsed in this case (WER 1.00), outputting "So So So So So." This tells us that averaged attention heatmaps are a lossy summary - similar global structure does not guarantee usable lyrics.

### Jazz: A Regression Case

![Cross-Attention Maps - Jazz](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_jazz.png)

The **Baseline** jazz attention produced reasonably good lyrics (WER 0.26). The **LoRA** version regressed to WER 0.42 and truncated the output earlier.

Visually subtle changes, meaningful output differences - sometimes improvements, sometimes regressions. The LoRA adapter is clearly changing how the DiT uses conditioning, but the effect is strongly genre-dependent.

---

## Experiment 2: Attention Sharpness Regularization - The Deliberate Failure

### The Theory

After looking at the attention maps, I had an idea. One known problem with the base model is attention dilution - the cross-attention spreads too evenly across all 769 conditioning tokens instead of focusing sharply. What if I could directly penalize this during training?

Entropy measures how "spread out" a probability distribution is. A sharp distribution (most probability on one token) has low entropy. A uniform distribution has high entropy. So I added an entropy penalty to the training loss:

```
Total Loss = Flow Matching Loss + λ × Mean Cross-Attention Entropy
```

The idea: by penalizing high-entropy (diffuse) attention, the model should learn to be more "decisive" about which conditioning tokens matter at each time step.

I called this Attention Sharpness Regularization (ASR). I set λ=0.02 and retrained the LoRA adapter from scratch with this additional loss term.

### What Actually Happened

| Model | WER (%) | CER (%) | AAS (DiT) |
|-------|---------|---------|-----------|
| Baseline | 61.03 | 51.91 | 0.1182 |
| + LoRA | 46.51 | 39.08 | 0.1498 |
| **+ LoRA + ASR (λ=0.02)** | **89.21** | **77.73** | **0.0595** |

It made results much worse overall. WER went from 46.5% (LoRA) to 89.2% (LoRA+ASR). The attention alignment score dropped by about 60%. The model often stopped producing coherent output.

Looking at the actual outputs across 30 genres:

- **6 out of 30** produced silence or instrumental-only output (Whisper returned empty strings or 🎵)
- **17 out of 30** landed at WER = 1.00, and **3 out of 30** exceeded 1.00 due to long hallucinated outputs
- Only **3 out of 30** were below WER 0.50 (Latin Pop 0.0556, New Wave 0.0556, Punk 0.2174)

Some highlights from the hallucination category:
- **Folk** (WER 1.15, should have been "Walking down the morning road"): "Oh, for a night, he called, as his light, still we see so many, as his glory, so, as his torch, so many."
- **Blues** (WER 1.27): "The death of the living, the death of the living. We need to bless you, make it better for us. We need to thank you, and thank you Lord."
- **Tropical House** (WER 1.45): "...every moment feels so sweet Little light, little light, little light..."
- **Lullaby** (WER 1.00): "Thank you."
- **Ambient** (WER 1.00): "2-期煎零ı"

The model frequently fell into degenerate behavior: empty output, transcript-style boilerplate, repetitive loops, or unrelated text.

No genre reached WER 0.00 in this run. The best results were **Latin Pop** and **New Wave** (both WER 0.0556), with **Punk** also relatively strong (WER 0.2174).

### Why It Failed: Attention Collapse

Go back and look at the rightmost panel ("Lora Asr") in any of the four heatmap figures above.

In every single one, the LoRA+ASR attention map shows the same pattern: **one or two extremely bright horizontal bands, with everything else completely dark**. Every audio frame, at every point in time, attends to the exact same one or two tokens. Usually the structural tag `[Verse]` or a newline character.

This is **attention collapse**. The entropy penalty told the model "be sharp - don't spread your attention." The model complied by finding the easiest way to minimize entropy: just always attend to the same token. One token gets all the probability mass → entropy is minimized → the regularization loss is happy.

But this is a degenerate solution. The model isn't attending to the *right* token at each time - it's attending to the *same* token at all times. The alignment signal is completely destroyed.

Compare this to the Baseline and LoRA panels in the same plots: their attention isn't perfectly diagonal, but there's temporal variation. Different frames attend to different tokens to different degrees. That imperfect, messy pattern is actually doing useful work. The ASR penalty largely removed it.

### What This Teaches Us

**1. Cross-attention in diffusion transformers is fragile.**

In language models, attention patterns are fairly robust to regularization. In a diffusion transformer, the cross-attention is the *entire mechanism* through which the conditioning signal (lyrics, caption, timbre) reaches the denoising process. Perturb it too much and the model loses access to its conditioning entirely.

**2. "Sharp" does not mean "correct."**

The entropy penalty achieves its mathematical objective perfectly - the attention distributions become extremely sharp. But sharp attention to the wrong token is worse than diffuse attention across many tokens. Diffuse attention at least lets some correct signal leak through. Collapsed attention blocks everything except the one token it fixated on.

**3. The base model's "diffuse" attention might be a feature, not a bug.**

The turbo-distilled ACE-Step model uses only 8 denoising steps (compressed from 50). With so few steps, each step has to do a lot of work. Spreading attention across multiple tokens gives the model robustness - it can average contributions from several "approximately right" tokens rather than having to nail the exactly right one every time. The entropy penalty removes this safety net.

**4. λ=0.02 was probably too aggressive, but the effective range might be vanishingly small.**

With λ=0.02 causing complete collapse, maybe λ=0.001 would work. But each evaluation round takes ~45 minutes (generate 30 audio clips, transcribe with Whisper, compute alignment). A proper sweep wasn't feasible in the time I had.

**5. Different layers probably need different treatment.**

The entropy penalty was applied uniformly across all 24 DiT layers. But early layers probably serve different roles than deep layers. Early layers might need broad attention for global style conditioning ("this should sound like jazz"), while deep layers might benefit from sharper attention for phonetic details. A blanket penalty ignores this structure.

---

## Stepping Back: What Did We Actually Learn?

### The Big Surprise

The most surprising result from all of this is that lyric intelligibility improved at all. The LoRA adapter was trained on audio paired with nothing but `[verse]` as lyrics - no words, no phonemes, no alignment signal. Yet at inference time, when given real lyrics, the adapted model produced clearer words.

This suggests something about how ACE-Step's vocal rendering works: the quality of the phonetic output depends not just on the lyric conditioning, but on the overall quality of the vocal rendering pathway. By fine-tuning on clean studio vocals with descriptive prompts, the adapter improved the general vocal clarity of the DiT decoder. The existing (base model) lyric-to-audio alignment mechanism then worked better simply because the "audio generation" side of the equation was producing cleaner output.

It's like the difference between a singer who knows the words but has a bad microphone, versus the same singer with a good microphone. The lyrics don't change, but they become easier to understand.

### About LoRA Fine-Tuning on Diffusion Models

1. **Even "wrong" fine-tuning data can help.** The training data had no lyric alignment whatsoever, yet lyric intelligibility improved. The model extracted useful information (what clean singing sounds like) from imperfect data (no word-level supervision). This is both encouraging and cautionary - it means LoRA adapts the model in ways that aren't always predictable from the training data alone.

2. **Where you apply LoRA matters more than how much.** Both self-attention and cross-attention adapters together work better than either alone. Adding MLP adapters on top (tripling the parameter count) doesn't help. The attention projections are the leverage point.

3. **Fine-tuning on a narrow distribution creates trade-offs.** The GTSinger data is all clean acapella studio recordings. The LoRA adapter improved vocal-forward genres (folk, jazz, gospel) at the cost of genres that were already well-served (hip-hop, EDM, rock). You're shifting the model's distribution, not expanding it.

4. **Rank 8 is a sweet spot for this task.** Rank 4 was too constrained (only improved 3/6 test cases). Rank 16 was equivalent to rank 8 at double the cost. The effective dimensionality of the update is apparently around 8.

### About Cross-Attention in Music Diffusion Models

1. **The attention pattern is not cleanly diagonal.** Even in successful generations, cross-attention doesn't look like a neat monotonic alignment. There are "attractor" tokens (structural tags, newlines) that get attention at all time positions. The useful signal is in the subtle, distributed pattern underneath.

2. **Visual similarity in attention maps can hide functional differences.** The Baseline and LoRA maps often look very similar when averaged across layers and heads, but the outputs are very different (silence vs. three lines of correct lyrics). The meaningful changes happen at individual layers and heads (which is kinda obvious, but just wanted to reiterate it here).

3. **The conditioning pathway is a single point of failure.** All lyric information reaches generation through cross-attention. If cross-attention fails, lyrics fail. This makes it both the most impactful intervention point and the most fragile one - as the ASR experiment demonstrated.

### About Negative Results

The ASR experiment is a failure in terms of improving lyrics, but it's one of the most informative results in the project:

- Naive entropy regularization on cross-attention in diffusion transformers causes attention collapse
- The attention distributions in turbo-distilled models encode a delicate balance that shouldn't be perturbed globally
- "Make the attention sharper" sounds like it should help, but the model finds the cheapest way to satisfy the penalty (collapse to one token) rather than the useful way (attend to the right token at each time)

Sometimes the failed experiment teaches you more than the successful one.

---

## Summary of Part 1 Experiments

| Experiment | What Changed | WER | CER | Verdict |
|-----------|-------------|-----|-----|---------|
| Baseline ACE-Step v1.5 | Nothing | 61.03% | 51.91% | Lyrics bad in most genres |
| + LoRA (r=8, 100 epochs) | Fine-tuned on clean vocal audio with descriptive prompts (no word lyrics) | 46.51% | 39.08% | Improved vocal clarity → improved lyric intelligibility as a side effect. Regressions in some genres. |
| + LoRA + ASR (λ=0.02, 50 epochs) | Added entropy penalty on cross-attention | 89.21% | 77.73% | Major failure: attention collapse and repetitive/off-target outputs. |

| LoRA Ablation | Configuration | Result (6 test cases) |
|--------------|---------------|----------------------|
| Rank 4 | alpha=8, 2.8M params | 3/6 clear lyrics |
| **Rank 8** | **alpha=16, 5.5M params** | **5/6 clear lyrics** |
| Rank 16 | alpha=32, 11M params | 5/6 clear lyrics |
| Cross-attn only | 2.8M params | 3/6 clear lyrics |
| Self-attn only | 2.8M params | 2/6 clear lyrics |
| **Both** | **5.5M params** | **5/6 clear lyrics** |
| Both + MLP | 18.2M params | 4/6 clear lyrics |

*Note: the ablation table above comes from a separate 6-case development sweep and is not derived from `eval_output_5way/results.csv`.*

---

## Reproducing This

Everything runs on a single consumer GPU. Each LoRA adapter is 22MB. Training takes about 3 hours per variant, and all four variants can be trained in parallel since they are pretty small in size.

---

## Part 2: Training With Actual Lyrics - The Results

The training runs finished. Time to see if our predictions were right.

### What Changed in the Data

After Part 1 revealed the empty-lyrics bug, I fixed the preprocessing pipeline to extract word-level lyrics from GTSinger's phonetic annotations. The corrected dataset has 2,389 clips - every training sample now pairs the audio with real, properly formatted lyrics:

```
[verse]
there's a river full of memory sleep my darling safe and sound
```

Each `.wav` file has a corresponding `.txt` with the actual words being sung. The captions and genre tags remained unchanged - the *only* difference is the lyrics field going from `[verse]` (empty) to `[verse]\nactual words being sung...`.

### What the Cross-Attention Learns Differently

Before looking at numbers, it's worth understanding what changes structurally when the lyrics field is no longer empty. Refer back to the architecture diagram from Part 1.

**Part 1 (no lyrics):** Every training sample had `lyrics = "[verse]"`. The LyricEncoder processed a near-empty input - only the first few tokens (`[`, `verse`, `]`, `\n`) had meaningful content, the rest were padding. When the Condition Encoder packed everything into the 769-token sequence, the lyric portion was essentially a block of padding with a few tag tokens at the front.

```
Part 1 - What the cross-attention sees during training:

Packed conditioning [769 tokens]:
┌─────────────────────┬───┬─────────────────────────────────┐
│ Lyric tokens (~512) │ T │ Caption tokens (~256)           │
│ [verse] \n PAD PAD  │ i │ "clean female alto vocals,      │
│ PAD PAD PAD PAD ... │ m │  breathy singing technique..."  │
│ PAD PAD PAD PAD ... │ b │                                 │
│ (almost all padding)│ r │ (rich, descriptive content)     │
└─────────────────────┴───┴─────────────────────────────────┘
                      ↑
           Cross-attention learns to mostly
           ignore the lyric tokens (padding)
           and focus on the caption tokens
```

The LoRA adapters on the cross-attention K and V projections learned to represent these near-empty lyric tokens in a way that supported cleaner vocal rendering - because the *audio* was clean studio vocals and the *caption* described clean vocals. But the LoRA had no reason to learn anything about word-level content.

**Part 2 (with lyrics):** Now every sample has `lyrics = "[verse]\nwaking up i see that everything is okay..."`. The LyricEncoder processes a sequence where ~20-50 tokens are real words, and the attention mask tells the model which tokens carry content.

```
Part 2 - What the cross-attention sees during training:

Packed conditioning [769 tokens]:
┌──────────────────────────────┬───┬──────────────────────────┐
│ Lyric tokens (~512)          │ T │ Caption tokens (~256)    │
│ [verse] \n waking up i see   │ i │ "clean female alto       │
│ that everything is okay PAD  │ m │  vocals, breathy singing │
│ PAD PAD PAD PAD ...          │ b │  technique..."           │
│ (real words + padding)       │ r │                          │
└──────────────────────────────┴───┴──────────────────────────┘
                      ↑
           Cross-attention must now learn to
           attend to the RIGHT lyric tokens
           at the RIGHT time - flow matching
           loss penalizes misalignment implicitly
```

This changes the learning signal fundamentally:

- The **K and V LoRA adapters** see *different lyric content for every sample*. They must learn to represent word tokens as keys and values that carry phonetic information.
- The **Q LoRA adapters** now have a reason to learn temporal alignment - at time t in the audio, queries should attend to the lyric tokens being sung at time t. The flow matching loss provides implicit alignment supervision: wrong lyric tokens at time t → wrong predicted audio → higher loss.
- The **self-attention LoRA adapters** benefit indirectly - with proper lyric conditioning flowing through cross-attention, self-attention has better information for maintaining coherence between what's being sung and the surrounding audio context.

### Our Predictions (Before Seeing Results)

We predicted:
- **Lower WER/CER** than both baseline and no-lyrics LoRA, because the model would have both cleaner vocals AND better lyric-to-audio alignment
- **Higher AAS**, because cross-attention patterns should become more structured
- **Fewer genre regressions**, because lyric alignment should counterbalance the vocal-style bias from GTSinger
- **ASR partial recovery** - with real lyrics to align *to*, the entropy penalty shouldn't collapse to padding tokens

Time to see how wrong (and right) we were.

---

### The Five-Way Head-to-Head

I ran two new training configurations on the corrected data - LoRA with lyrics (rank 8, alpha 16, lr 1e-4, 55 epochs) and LoRA with lyrics + ASR (λ=0.02, 55 epochs) - then re-evaluated all five models across the same 30-genre test set under identical conditions (same prompts, same random seeds) for a fair comparison.

| Model | WER (%) ↓ | CER (%) ↓ | AAS (DiT) ↑ | AAS (LM) ↑ |
|-------|-----------|-----------|-------------|-------------|
| Baseline ACE-Step v1.5 | 61.03 | 51.91 | 0.118 | 0.181 |
| + LoRA (no lyrics) | **46.51** | 39.08 | 0.150 | 0.181 |
| + LoRA + ASR (no lyrics) | 89.21 | 77.73 | 0.060 | 0.149 |
| + LoRA with lyrics | 47.14 | **38.06** | 0.133 | **0.197** |
| + LoRA with lyrics + ASR | 64.23 | 47.52 | **0.166** | 0.175 |

*All five models evaluated on 30 genres with identical random seeds for fair comparison from the same `eval_output_5way` run used above.*

First reaction: **adding real lyrics to training barely changed the average WER.** LoRA with lyrics (47.14%) sits within 1 percentage point of LoRA without lyrics (46.51%). The CER improved by about 1 point (38.06% vs 39.08%). Not exactly the dramatic improvement we predicted.

But hold on - before declaring the experiment a wash, let's look at where those averages come from.

### Same Average, Completely Different Behavior

The averages are hiding something important. When you break down per-genre results, the two LoRA models are clearly **complementary**: no-lyrics LoRA has 10 unique wins, while lyrics-trained LoRA has 8 unique wins (plus 3 ties), and they are often strong on different genres.

**Where lyrics training rescued previously failed generations:**

| Genre | LoRA (no lyrics) | LoRA (with lyrics) | What Whisper Heard (with lyrics) |
|-------|------------------|--------------------|--------------------------------|
| Soul | 0.28 | **0.00** | "Take my hand and walk with me / Down the road we used to know / Every river meets the sea / Every seed will start to grow" - every single word perfect |
| Disco | 1.00 (🎵) | **0.13** | "Saturday is here at last / Put your dancing shoes on tight / Live your ways in the past / We are gonna dance all night" |
| Lullaby | 1.00 (silence) | **0.18** | "Close your eyes and drift away / Stars are shining just for you / Tomorrow brings another day / Full of wonder, romance" |
| R&B | 1.00 ("So So So So So") | **0.38** | "Close your eyes / Let the mirror take us home / Breathe above the skies" - structure right, some words off |
| Hip-hop | 1.00 ("adult rhymes") | **0.41** | "Walking through the city streets, head held high above the crowd. Javier makes me stand, stand a little proud." |
| Jazz | 0.42 | **0.21** | "It falls on empty streets / Silver shadows come alive / Every whisper softly rings / Remember of you and I" |

These are genres where the Part 1 LoRA either failed outright (silence/instrumental/degenerate output) or underperformed. The lyrics-trained model rescued many of them. Soul improved from WER 0.28 to literally perfect (0.00). Disco went from complete silence to a recognizable dance track with nearly all lyrics correct.

**Where lyrics training broke what was working:**

| Genre | LoRA (no lyrics) | LoRA (with lyrics) | What Went Wrong |
|-------|------------------|--------------------|----------------|
| New Wave | **0.00** (perfect) | 1.00 | From perfect lyrics to silence - 🎵 |
| Rock | **0.17** | 1.00 | From "Standing on the edge of town, Staring for the sun to rise" to "CBD Black Bloomberg Sour Full Appearance" |
| Power Ballad | **0.09** | 0.52 | From near-perfect to truncated output |
| EDM | **0.27** | 0.82 | From mostly correct to garbled with inserted "La la la la" |
| Latin Pop | **0.17** | 0.72 | From clean lyrics to repetitive fragments: "The air The air More burns through the air" |
| Synthwave | **0.00** (perfect) | 0.24 | From perfect to "The lights are flashing bright" - close but no longer exact |

The rock result is worth highlighting. The no-lyrics LoRA produced reasonably clear singing: "Standing on the edge of town, Staring for the sun to rise, Holding on to what is mine, Staring for the open sky" - WER 0.17, minor word substitutions but clearly the right lyrics. The lyrics-trained LoRA produced "CBD Black Bloomberg Sour Full Appearance" - complete gibberish with no meaningful overlap. Same genre, same prompt, same seed, different adapter.

### Why the Two LoRA Models Help Different Genres

This complementary behavior has a satisfying explanation if you think back to what each adapter learned during training.

**LoRA (no lyrics)** was trained on clean studio vocals with `[verse]` as lyrics. The cross-attention LoRA adapters learned to extract vocal quality information from the *caption* ("clean female alto vocals, breathy singing technique...") and project it into better vocal rendering. At inference time, when given real lyrics, the improved vocal rendering helps the base model's existing (imperfect) lyric alignment work better. Think of it as upgrading the microphone - the singer already knows the words, you just made them clearer.

This approach succeeds in genres where the base model already has *some* vocal rendering capability that just needs to be cleaner: folk, pop, rock, synthwave, bossa nova. It fails in genres where the base model's vocal rendering is fundamentally broken - the model never attempts vocals in the first place.

**LoRA (with lyrics)** was trained on the same audio with real lyrics aligned to each recording. The cross-attention LoRA adapters learned something different: the K and V projections learned to represent *word tokens* as keys and values that carry phonetic information; the Q projections learned to formulate queries that match audio time positions to corresponding lyric tokens. The flow matching loss provided implicit alignment supervision.

This approach succeeds in genres where the base model's vocal rendering was so broken it produced silence or degenerate output. The lyric alignment provides an *additional activation signal* - the model learned that "when there are real words in the lyrics, you should sing them." This is exactly the capability the no-lyrics adapter never acquired.

It fails in genres where the no-lyrics adapter had already found a good solution, because the lyrics training introduces a different bias that can conflict with genre-appropriate vocal styling.

Think of it this way:
- **No-lyrics LoRA:** "I know what clean singing *sounds* like" → improves quality of existing vocals
- **With-lyrics LoRA:** "I know what singing *words* looks like" → activates vocal rendering where it was missing

Neither is strictly better. They're optimizing for different aspects of the same problem.

### The Best Model Depends on the Genre

Tallying which model achieves the lowest WER for each of the 30 genres:

| Best Model | Genres Won | Notable Examples |
|-----------|-----------|----------|
| Baseline | 3 unique (+2 ties) | EDM (0.00), R&B (0.14), Singer-songwriter (0.11) |
| LoRA (no lyrics) | 10 unique (+3 ties) | Pop (0.00), Synthwave (0.00), Bossa Nova (0.00), Rock (0.17), New Wave (0.00) |
| LoRA with lyrics | 8 unique (+3 ties) | Soul (0.00), Disco (0.13), Lullaby (0.18), Jazz (0.21), Grunge (0.05) |
| LoRA+lyrics+ASR | 4 unique | Punk (0.04), Indie (0.05), Celtic (0.28), Blues (0.68) |
| LoRA+ASR (no lyrics) | 1 unique | Latin Pop (0.06) - one of its few non-collapse results |

No single model wins everywhere. If you could pick the best adapter per genre, the hypothetical "oracle ensemble" would likely outperform any individual model. This suggests the path forward isn't "one LoRA to rule them all" but rather a mixture or routing system that selects the right adapter based on genre characteristics.

### A Closer Look: Transcription Showdowns

Some of the most revealing side-by-side comparisons across all five models:

**Soul - The Perfect Score**

Ground truth: *"Take my hand and walk with me / Down the road we used to know / Every river meets the sea / Every seed will start to grow"*

| Model | Transcription | WER |
|-------|--------------|-----|
| Baseline | *(silence)* | 1.00 |
| LoRA (no lyrics) | "Take my hand and you walk with me / Down the road we used to know / Every rhythm meets the sea / Every sea" | 0.28 |
| LoRA+ASR | "And I visit the answer / And I meet the sea / And we see we'll start to grow" | 0.76 |
| **LoRA with lyrics** | **"Take my hand and walk with me / Down the road we used to know / Every river meets the sea / Every seed will start to grow"** | **0.00** |
| LoRA+lyrics+ASR | "Take my hand and then walk with me / Down the road we used to know / Every river meets the sea / It's the sea that will start to grow Yeah, to grow" | 0.32 |

A clean progression from silence → partial → perfect. The lyrics-trained model nailed every word. Even the ASR variant got close.

**Punk - Where Sharpness Actually Helps**

Ground truth: *"Burning down the walls tonight / Nothing left for us to lose / Standing up to join the fight / We will never pay our dues"*

| Model | Transcription | WER |
|-------|--------------|-----|
| Baseline | "We will never pay our dues." | 0.78 |
| LoRA (no lyrics) | "Burning out the walls tonight, nothing left for us to lose / Standing up to join the fight, we will never back our doom" | 0.22 |
| LoRA+ASR (no lyrics) | "Turning down the wall tonight, nothing left for us to lose / Standing up to join some fight, we will never pay our dues" | 0.22 |
| LoRA with lyrics | "I'm turning down the walls tonight. Nothing left for us to lose. / Standing up to join the fight. We'll pay our dues." | 0.39 |
| **LoRA+lyrics+ASR** | **"Burning down the walls tonight / Nothing left for us to lose / Standing up to join the fight / We will never pay your dues"** | **0.04** |

Nearly perfect - only "your" instead of "our." Punk's strong rhythmic delivery aligns well with the forced sharp attention from the ASR penalty. When the music has a clear, percussive beat and the words land on strong rhythmic positions, sharpened attention is an asset rather than a liability.

**Pop - When Everything Goes Wrong For One Model**

Ground truth: *"I can feel the rhythm tonight / Dancing underneath the stars / Every heartbeat burning bright / Nothing in the world is ours"*

| Model | WER |
|-------|-----|
| Baseline | 0.85 |
| **LoRA (no lyrics)** | **0.00** (perfect) |
| LoRA+ASR (no lyrics) | 0.55 |
| LoRA with lyrics | 0.05 |
| LoRA+lyrics+ASR | **2.10** (!!!) |

The LoRA (no lyrics) model achieved a perfect score on the evaluation seed (43). To sanity-check this, I ran three additional seeds (44, 45, 46) - two were also perfect, one had minor substitutions. The mean WER across four seeds was 0.075, so the strong result is real but not universally zero. The lyrics+ASR variant produced severe hallucination - WER exceeding 2.0 means it generated more than twice the words in the ground truth, none of them correct: *"Like and thinning will be fine / Dog and dog's in line / Chin-na-ta-tine / Lines on your door / Turn it this time / First off, every happy, happy day..."* - a much longer, off-target transcript than the original lyrics. The sharp attention penalty locked the model into lyric tokens but it couldn't decode them, so it kept generating incorrect text.

---

## LoRA + ASR With Real Lyrics: The Partial Redemption

Remember, in Part 1, attention sharpness regularization (ASR) with no lyrics was our worst failure - WER rose to 89%, the model often generated repetitive outro-like text instead of proper lyrics, and the attention collapsed to single tokens. We hypothesized that ASR failed partly because there were no real lyrics to align *to*.

The results partially confirm this hypothesis.

| Metric | LoRA+ASR (no lyrics) | LoRA+ASR (with lyrics) | Change |
|--------|---------------------|----------------------|--------|
| WER | 89.21% | 64.23% | -25 points |
| CER | 77.73% | 47.52% | -30 points |
| AAS (DiT) | 0.060 | **0.166** | Nearly 3x improvement |
| Cases with WER >= 1.0 | 20 of 30 genres | 4 of 30 genres | -16 genres |

The with-lyrics ASR model is still worse than the simpler LoRA variants on average WER, but it's no longer a total collapse. More importantly, it achieves the **highest attention alignment score of any model** (0.166) - its cross-attention patterns are more structured and temporally organized than any other variant, including the baseline.

### Why Best Alignment ≠ Best Lyrics

Here's what the attention metrics from the deep-analysis pipeline reveal:

| Metric | Baseline | LoRA | LoRA+ASR | LoRA+lyrics | LoRA+lyrics+ASR |
|--------|----------|------|----------|-------------|-----------------|
| Lyric Mass (folk) | 0.43 | 0.48 | 0.43 | **0.64** | **0.84** |
| Lyric Mass (pop) | 0.41 | 0.42 | 0.32 | 0.40 | **0.72** |
| Lyric Mass (jazz) | 0.34 | 0.38 | 0.30 | 0.37 | **0.71** |
| Lyric Mass (R&B) | 0.44 | 0.48 | 0.45 | 0.43 | **0.73** |
| Entropy (folk) | 0.71 | 0.73 | 0.59 | 0.54 | **0.48** |
| Entropy (pop) | 0.78 | 0.67 | 0.61 | 0.65 | **0.52** |
| Entropy (jazz) | 0.71 | 0.66 | 0.63 | 0.64 | **0.53** |
| Entropy (R&B) | 0.75 | 0.74 | 0.55 | 0.66 | **0.52** |

*Lyric Mass = fraction of cross-attention going to lyric tokens vs. caption/timbre tokens. Entropy = how spread the attention is across tokens (lower = sharper). Data from the deep-analysis pipeline using lyric-only renormalization.*

The lora+lyrics+ASR variant devotes **72-84% of its cross-attention to lyric tokens** - substantially more than any other model (30-64%). And its entropy is consistently the lowest, meaning its attention is the sharpest. The ASR penalty with real lyrics causes the model to:

1. **Focus on lyrics, not caption.** While other models split attention roughly 40/60 between lyrics and caption, the ASR+lyrics model overwhelmingly attends to the lyric tokens.
2. **Be decisive about which lyric token.** The low entropy means at each audio time position, the model commits to a few specific tokens rather than spreading attention thinly.

This produces the best attention alignment scores - but not the best WER. The model is doing a great job of *looking at the right lyric tokens at the right time*, but a poor job of *turning that attention into correct phonemes*. The value projections (V) and output projections (O) - which determine what information gets extracted from the attended tokens - apparently need more training to learn effective phonetic rendering under the aggressive ASR penalty.

It's like a student who learned to look at the right page in the textbook at the right time, but hasn't yet learned to read the words on the page. The attention is well-aligned; the decoding isn't.

The cases where it DOES work (punk at WER 0.04, indie at 0.05, tropical house at 0.15) tend to be genres with clear, rhythmic vocal delivery where the phonetic mapping is simpler. The cases where it fails badly (pop at WER 2.10) tend to be genres requiring more melodic, flowing vocals where the rigid attention pattern fights against the music's natural phrasing.

### The ASR Controlled Experiment

Putting the four LoRA variants in a 2×2 grid reveals the cleanest controlled experiment in this project:

| | No ASR (λ=0) | ASR (λ=0.02) |
|---|---|---|
| **No lyrics** | LoRA: WER 46.51% (improved) | LoRA+ASR: WER 89.21% (severe regression) |
| **With lyrics** | LoRA+lyrics: WER 47.14% (improved) | LoRA+lyrics+ASR: WER 64.23% (partial recovery) |

Without lyrics, ASR causes attention collapse - the model attends to padding tokens because that's the cheapest way to minimize entropy. With lyrics, ASR forces the model to attend sharply to real word tokens, which is actually useful - but λ=0.02 is still too aggressive for general use.

The key insight: **entropy regularization on cross-attention is not inherently bad, but it requires meaningful conditioning content.** You can't force the model to "pay attention" when there's nothing meaningful to attend to.

---

## Going Deeper: Per-Head Cross-Attention Analysis

The summary heatmaps average attention across all 24 layers and all 16 query heads (8 KV heads in the GQA scheme). As the R&B analysis hinted earlier - where a simpler-looking attention pattern produced better lyrics than a more structured one - that averaging hides important structure.

To dig deeper, we generated per-head attention maps for all five model variants across four genres (folk, pop, jazz, R&B) using the improved deep-analysis pipeline. This pipeline uses deterministic extraction at the schedule-consistent timestep, lyric-only renormalization to isolate the lyric-audio relationship, and shared color scales for fair comparison across variants.

### Folk - Per-Head Across All Five Variants

**Baseline:**

![Per-Head Attention - Folk Baseline](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_folk_baseline_per_head.png)

**LoRA (no lyrics):**

![Per-Head Attention - Folk LoRA](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_folk_lora_per_head.png)

**LoRA+ASR (no lyrics):**

![Per-Head Attention - Folk LoRA+ASR](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_folk_lora_asr_per_head.png)

**LoRA with lyrics:**

![Per-Head Attention - Folk LoRA with Lyrics](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_folk_lora_with_lyrics_per_head.png)

**LoRA with lyrics+ASR:**

![Per-Head Attention - Folk LoRA with Lyrics+ASR](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_folk_lora_with_lyrics_asr_per_head.png)

Even per-head (averaged across 24 layers), most heads show a single dominant horizontal band - one token capturing most of the attention at all time positions. This is consistent across all five model variants. Tokens like `\n`, `[`, or high-frequency words like "the" act as attention sinks that absorb probability mass regardless of audio position.

But the subtle differences between variants matter. The LoRA with lyrics folk per-head map shows slightly more variation in band structure compared to the no-lyrics LoRA - some heads have multiple bands or show transitions between tokens. The ASR+lyrics variant shows the most extreme concentration: nearly all probability mass in a few very sharp bands, consistent with its much higher lyric mass (0.84) and lower entropy (0.48).

### Pop - Per-Head Comparison

**Baseline:**

![Per-Head Attention - Pop Baseline](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_pop_baseline_per_head.png)

**LoRA (no lyrics)** (WER 0.00 on the evaluation seed; mean 0.075 across 4 seeds - consistently strong):

![Per-Head Attention - Pop LoRA](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_pop_lora_per_head.png)

**LoRA with lyrics:**

![Per-Head Attention - Pop LoRA with Lyrics](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_pop_lora_with_lyrics_per_head.png)

**LoRA with lyrics+ASR** (WER 2.10 - severe hallucination):

![Per-Head Attention - Pop LoRA with Lyrics+ASR](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_pop_lora_with_lyrics_asr_per_head.png)

The pop per-head maps show a clear progression. The LoRA variant (WER 0.00 on the eval seed, mean 0.075 across 4 seeds) shows structured head behavior - a few heads carry temporal patterns while others stay collapsed. The lyrics+ASR variant, which hallucinated heavily, shows the extreme version: attention mass concentrated in sharp bands with barely any temporal variation. The model was attending strongly to lyric tokens but couldn't decode them.

### Jazz and R&B - Per-Head Views

**Jazz Baseline:**

![Per-Head Attention - Jazz Baseline](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_jazz_baseline_per_head.png)

**Jazz LoRA with Lyrics:**

![Per-Head Attention - Jazz LoRA with Lyrics](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_jazz_lora_with_lyrics_per_head.png)

**R&B Baseline:**

![Per-Head Attention - R&B Baseline](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_rnb_baseline_per_head.png)

**R&B LoRA (no lyrics):**

![Per-Head Attention - R&B LoRA](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_rnb_lora_per_head.png)

**R&B LoRA with Lyrics:**

![Per-Head Attention - R&B LoRA with Lyrics](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures_codex/attention_rnb_lora_with_lyrics_per_head.png)

### The Sparse Alignment Signal

The per-layer-head statistics quantify what the figures show qualitatively. For each of the 384 layer-head combinations (24 layers × 16 heads), the deep-analysis pipeline computed a "diagonal score" measuring how well the attention path follows the temporal order of lyrics. Here are the best individual heads for folk:

| Variant | Best Head | Layer | Diagonal Score | Correlation |
|---------|-----------|-------|---------------|-------------|
| **LoRA with lyrics** | **Head 11** | **Layer 3** | **0.224** | **0.77** |
| Baseline | Head 4 | Layer 15 | 0.188 | 0.41 |
| LoRA+lyrics+ASR | Head 9 | Layer 5 | 0.165 | 0.82 |
| LoRA (no lyrics) | Head 10 | Layer 3 | 0.146 | 0.69 |
| LoRA+ASR (no lyrics) | Head 4 | Layer 22 | 0.090 | 0.16 |

The lyrics-trained model has the highest single-head diagonal score for folk (0.224, with 0.77 correlation to temporal order). And something interesting: this best head lives at **Layer 3** - one of the earliest DiT layers. In the baseline, the best alignment head is at Layer 15, deep in the network, while no-lyrics LoRA also peaks early at Layer 3. This suggests LoRA adaptation can shift strong alignment heads earlier, with lyrics-trained LoRA giving the strongest early-layer score in folk. This could be significant - early-layer alignment means the temporal structure is established before the deeper layers handle higher-level musical decisions.

Across all four genres:

| Genre | Baseline Best | LoRA Best | LoRA+lyrics Best | LoRA+lyrics+ASR Best |
|-------|-------------|-----------|-----------------|---------------------|
| Folk | 0.188 | 0.146 | **0.224** | 0.165 |
| Pop | 0.212 | **0.253** | 0.243 | 0.227 |
| Jazz | **0.265** | 0.157 | 0.222 | 0.160 |
| R&B | **0.363** | 0.173 | 0.172 | 0.186 |

The lyrics-trained model matches or exceeds the baseline's best diagonal scores in folk and pop, while jazz and R&B show the baseline maintaining higher scores. The pattern is inconsistent - further evidence that alignment is a fragile, genre-dependent property rather than a robust global capability.

### What Per-Head Analysis Teaches Us

The per-head analysis tells a consistent story across all five variants:

1. **Most heads are "sink" heads.** In all models, many layer-head combinations (especially in early layers 0-2) show `dominant_fraction = 1.0` - a single token gets ALL the attention at every time step. Common sink tokens: `\n`, `[`, and common words like "the."

2. **Alignment signal is sparse.** Only a handful of specific layer-head combinations show meaningful temporal movement. Out of 384 total combinations, typically fewer than 10 have diagonal scores above 0.1.

3. **Lyrics training doesn't eliminate sink heads - it makes the few alignment heads more effective.** The LoRA with lyrics model still has hundreds of collapsed heads, but the best heads show stronger temporal correlation.

4. **The model uses lyrics more like a global conditioning cue than a word-by-word temporal script.** Even the best alignment head (diagonal correlation 0.77) is far from the clean monotonic alignment you'd see in a well-trained TTS system. Cross-attention in the DiT is a conditioning mechanism, not an alignment mechanism.

---

## Updated Cross-Attention Heatmaps: All Five Models

Go back to the cross-attention heatmaps from earlier - now with all five panels visible.

### Folk - Five-Way Comparison

![Cross-Attention Maps - Folk (5 variants)](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_folk.png)

The **LoRA with lyrics** panel (fourth) shows a different attention distribution compared to LoRA without lyrics. There's more lyric mass (the lyric portion of the heatmap is brighter relative to the summary), consistent with the deep-analysis metrics showing lyric mass of 0.64 vs 0.48. The **LoRA+lyrics+ASR** panel (fifth) is the most striking - overwhelmingly concentrated on lyric tokens with sharp, distinct bands. This is what lyric mass of 0.84 looks like.

### Jazz - Five-Way Comparison

![Cross-Attention Maps - Jazz (5 variants)](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_jazz.png)

The lyrics-trained model improved jazz WER from 0.42 to 0.21. The heatmap shows why - the fourth panel has more distributed attention across lyric tokens compared to the third panel (LoRA+ASR, which is nearly uniform). The fifth panel (lyrics+ASR) shows extreme lyric focus but in a less structured way.

### R&B - Five-Way Comparison

![Cross-Attention Maps - R&B (5 variants)](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_rnb.png)

R&B tells the story of the complementary models. The second panel (LoRA without lyrics, WER 1.00 - "So So So So So") shows attention that looks reasonable at the aggregate level but completely fails at generation. The fourth panel (LoRA with lyrics, WER 0.38) has a different distribution that actually produces recognizable lyrics. Summary-level heatmaps continue to be unreliable predictors of output quality.

### Pop - Five-Way Comparison

![Cross-Attention Maps - Pop (5 variants)](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/figures/attention_pop.png)

The second panel (LoRA, WER 0.00) and fourth panel (LoRA with lyrics, WER 0.05) look qualitatively similar and both produce excellent results. The fifth panel (lyrics+ASR, WER 2.10) shows strongly concentrated attention - the model focuses heavily on lyrics but still generates off-target text.

---

## Complete Summary of All Experiments

| Experiment | Training Data | WER (%) | CER (%) | AAS (DiT) | Key Finding |
|-----------|--------------|---------|---------|-----------|-------------|
| Baseline ACE-Step v1.5 | - | 61.03 | 51.91 | 0.118 | Lyrics garbled or missing in ~60% of genres |
| + LoRA (no lyrics) | Clean vocals, `[verse]` only | **46.51** | 39.08 | 0.150 | Improved vocal clarity → better lyric intelligibility. Best at 10 genres (+3 ties). |
| + LoRA + ASR (no lyrics) | + entropy penalty | 89.21 | 77.73 | 0.060 | Severe attention collapse with repetitive/off-target output. |
| + LoRA with lyrics | Clean vocals + real lyrics | 47.14 | **38.06** | 0.133 | Nearly identical average, but wins 8 genres (+3 ties). Complementary to no-lyrics LoRA. |
| + LoRA with lyrics + ASR | + entropy penalty | 64.23 | 47.52 | **0.166** | Still weaker on WER, but no longer a collapse. Best attention alignment. 4 unique genre wins (punk, indie, celtic, blues). |

| LoRA Ablation (Part 1) | Configuration | Result (6 test cases) |
|-------------------------|---------------|----------------------|
| Rank 4 | alpha=8, 2.8M params | 3/6 clear lyrics |
| **Rank 8** | **alpha=16, 5.5M params** | **5/6 clear lyrics** |
| Rank 16 | alpha=32, 11M params | 5/6 clear lyrics |
| Cross-attn only | 2.8M params | 3/6 clear lyrics |
| Self-attn only | 2.8M params | 2/6 clear lyrics |
| **Both** | **5.5M params** | **5/6 clear lyrics** |
| Both + MLP | 18.2M params | 4/6 clear lyrics |

*Note: this ablation table comes from a separate 6-case development sweep and is not derived from `eval_output_5way/results.csv`.*

---

## What We Actually Learned: The Complete Picture

Four experiments, five model variants, 150 generated songs, and hundreds of per-head attention maps later - here's what this project taught us about LoRA fine-tuning on diffusion transformers.

### 1. Vocal Clarity and Lyric Alignment Are Separable Skills

The most fundamental finding: LoRA without lyrics improved vocal clarity (the model sings more clearly), while LoRA with lyrics improved lyric alignment (the model sings the right words at the right time). These are different skills, and they help in different situations.

Vocal clarity helps when the model is already trying to sing but producing garbled output - pop, rock, synthwave. Lyric alignment helps when the model isn't singing at all - it provides an activation signal that says "there are words here, sing them" - soul, disco, lullaby.

This separation wasn't obvious ahead of time. You might expect that training with aligned lyrics would subsume the benefits of training without them. But the lyric alignment training introduces a different optimization pressure on the cross-attention weights, leading to a different (not strictly better) adapter.

### 2. The Same LoRA Architecture Can Learn Very Different Things

Both adapters have identical architecture - rank 8, alpha 16, 5.5M parameters, same target modules. The only difference is the training data's lyrics field. Yet they produce complementary models that succeed and fail on entirely different genre sets.

This tells us that a rank-8 update to the cross-attention projections is expressive enough to encode *qualitatively different* behaviors: "make the voice cleaner" versus "align the words to audio timing." The low-rank bottleneck doesn't force both training objectives into the same subspace - there's apparently enough representational capacity at rank 8 for these to be distinct solutions.

### 3. Entropy Regularization Needs Content To Regularize

The 2×2 ASR experiment provides the cleanest controlled result:

| | No ASR | ASR (λ=0.02) |
|---|---|---|
| **No lyrics** | Improved (46.51%) | Catastrophic (89.21%) |
| **With lyrics** | Improved (47.14%) | Partial recovery (64.23%) |

Without lyrics, ASR causes attention collapse to padding tokens. With lyrics, ASR forces attention to real word tokens - useful in principle, still too aggressive at λ=0.02. **Entropy regularization on cross-attention requires meaningful conditioning content to work.**

### 4. Attention Alignment ≠ Output Quality

The lyrics+ASR model has the best AAS of any variant (0.166) yet worse WER (64.23%) than both simpler LoRA models (~47%). This clearly shows that structured attention patterns don't automatically produce correct lyrics.

The attention tells the model *where to look*. Turning that information into correct phonemes requires the value and output projections to encode useful phonetic representations - and those apparently need more training under an aggressive entropy penalty. The model learned *where* to look before it learned *what to do* with what it found.

### 5. Cross-Attention Uses Lyrics as a Global Cue, Not a Script

Across all five variants and all per-head analysis, the model treats lyrics more like a **global conditioning signal** than a word-by-word temporal guide. Most attention heads collapse to sink tokens. Only a sparse subset shows temporal movement. Even the best individual heads achieve 0.77 correlation with temporal lyrics order - strong but far from the clean monotonic alignment in a TTS system.

This suggests that truly solving lyric intelligibility in music diffusion models may require architectural changes - explicit alignment modules, duration predictors, or phoneme-level conditioning - rather than better training data or regularization alone. The cross-attention mechanism in the DiT is powerful but wasn't designed for precise temporal text-audio alignment.

### 6. No Single Adapter Wins Everything

The hypothetical "oracle" that picks the best model per genre would likely outperform any individual variant. The practical implication: build a routing mechanism that selects the right adapter based on genre or prompt characteristics, or explore weight-space merging techniques (adapter arithmetic at different scales) to capture complementary strengths in a single model.

---

## Listen For Yourself: Audio Examples

Numbers and heatmaps only tell half the story. Below are generated audio samples for pop across all five model variants, using the same prompt and lyrics from the evaluation. Put on headphones and try to follow the lyrics as you listen.

Note: these audio clips were generated in a separate run from the main evaluation. Diffusion models are sensitive to CUDA nondeterminism, so even with the same seed the outputs differ between runs. The WER values below are from Whisper transcription of *these specific audio files*, not from the main evaluation tables above.

**Ground truth lyrics (Pop):** *"I can feel the rhythm tonight / Dancing underneath the stars / Every heartbeat burning bright / Nothing in the world is ours"*

| Variant | WER | Whisper heard | Audio |
|---------|-----|---------------|-------|
| Baseline | 0.05 | "...Nothing in the world is odd" | [pop_baseline.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/audio_examples/pop_baseline.flac) |
| LoRA (no lyrics) | 0.10 | "Every heart beat burning bright" (minor split) | [pop_lora.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/audio_examples/pop_lora.flac) |
| LoRA+ASR (no lyrics) | 1.20 | "I can feel your sacrifice..." (stuck in loop) | [pop_lora_ashr.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/audio_examples/pop_lora_ashr.flac) |
| LoRA with lyrics | 0.15 | "Dancing in beneath the stars" | [pop_lora_with_lyrics.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/audio_examples/pop_lora_with_lyrics.flac) |
| LoRA+lyrics+ASR | 0.55 | "And I can feel the resultant tonight..." | [pop_lora_with_lyrics_ashr.flac](https://github.com/Karthik-Ragunath/learn-through-songs-blog/raw/main/blog/audio_examples/pop_lora_with_lyrics_ashr.flac) |

These samples illustrate the stochasticity of diffusion generation - the *ranking* between models can shift across runs. The aggregate statistics over 30 genres in the tables above are more reliable than any single sample. These audio clips are best used to get a qualitative feel for each model's vocal character rather than as definitive WER benchmarks.

---

*This started as a random experiment on a Friday night - throw some clean vocal data at a music generation model and see what happens. A preprocessing bug turned it into a natural ablation study. A failed entropy penalty taught us more about cross-attention dynamics than any of the successes. That's kind of how life rolls too!*
