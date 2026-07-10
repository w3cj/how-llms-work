/** Displays Word2Vec training progress and results: corpus stats, epoch losses, learned embeddings, nearest neighbors, pairwise similarity, and vector analogies. */
import clsx from "clsx";
import styles from "./styles.module.css";

export type InitData = {
  vocabSize: number;
  sentenceCount: number;
  embeddingDim: number;
  windowSize: number;
  totalPairs: number;
};

export type EpochData = { epoch: number; loss: number };
export type WordEmbedding = { word: string; vector: number[] };
export type Neighbor = { word: string; nearest: { word: string; score: number }[] };
export type SimilarityPair = { a: string; b: string; score: number };
export type Analogy = { query: string; result: string; score: number };

type Props = {
  init?: InitData;
  epochs: EpochData[];
  embeddings?: WordEmbedding[];
  neighbors?: Neighbor[];
  similarities?: SimilarityPair[];
  analogies?: Analogy[];
  warnings?: string[];
};

function lossClass(loss: number) {
  if (loss < 1.0)
    return styles.lossLow;
  if (loss > 5.0)
    return styles.lossHigh;
  return "";
}

function scoreClasses(score: number) {
  if (score >= 0.5)
    return { bar: styles.barFillHigh, text: styles.scoreHigh };
  if (score < 0.3)
    return { bar: styles.barFillLow, text: styles.scoreLow };
  return { bar: "", text: "" };
}

export function TrainEmbedResult({ init, epochs, embeddings, neighbors, similarities, analogies, warnings }: Props) {
  return (
    <div class="vstack">
      {init && (
        <>
          <div class={styles.label}>Corpus</div>
          <div class={styles.config}>
            <div class={styles.configItem}>
              sentences
              {" "}
              <span class={styles.configValue}>{init.sentenceCount}</span>
            </div>
            <div class={styles.configItem}>
              vocab
              {" "}
              <span class={styles.configValue}>{init.vocabSize}</span>
            </div>
            <div class={styles.configItem}>
              dimensions
              {" "}
              <span class={styles.configValue}>{init.embeddingDim}</span>
            </div>
            <div class={styles.configItem}>
              window
              {" "}
              <span class={styles.configValue}>{init.windowSize}</span>
            </div>
            <div class={styles.configItem}>
              training pairs
              {" "}
              <span class={styles.configValue}>{init.totalPairs}</span>
            </div>
          </div>
        </>
      )}

      {epochs.length > 0 && (
        <>
          <div class={styles.label}>{embeddings ? "Training" : "Training..."}</div>
          <div class={styles.epochList}>
            {epochs.map((e, i) => (
              <div key={i} class={styles.epochRow}>
                <span class={styles.epochNum}>
                  epoch
                  {" "}
                  {e.epoch}
                </span>
                <span class={lossClass(e.loss)}>
                  loss
                  {" "}
                  {e.loss.toFixed(6)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {warnings && warnings.length > 0 && (
        <div class={styles.warnings}>
          {warnings.map((w, i) => (
            <div key={i} class={styles.warning}>{w}</div>
          ))}
        </div>
      )}

      {embeddings && embeddings.length > 0 && (
        <>
          <div class={styles.label}>Learned Embeddings</div>
          <div class={styles.embeddings}>
            {embeddings.map((e, i) => (
              <div key={i} class={styles.embedding}>
                <div class={styles.embeddingText}>{e.word}</div>
                <div class={styles.vector}>
                  [
                  {e.vector.join(", ")}
                  ]
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {neighbors && neighbors.length > 0 && (
        <>
          <div class={styles.label}>Nearest Neighbors</div>
          <div class={styles.embeddings}>
            {neighbors.map((n, i) => (
              <div key={i} class={styles.neighborGroup}>
                <div class={styles.neighborWord}>{n.word}</div>
                <div class={styles.neighborList}>
                  {n.nearest.map((nb, j) => (
                    <div key={j} class={styles.neighborItem}>
                      <span class={styles.neighborName}>{nb.word}</span>
                      <span class={clsx(styles.neighborScore, nb.score >= 0.5 && styles.neighborScoreHigh)}>
                        {nb.score.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {similarities && similarities.length > 0 && (
        <>
          <div class={styles.label}>Pairwise Similarity</div>
          <div class={styles.similarities}>
            {similarities.sort((a, b) => b.score - a.score).map((s, i) => {
              const cls = scoreClasses(s.score);
              return (
                <div key={i} class={styles.similarity}>
                  <span class={styles.similarityPair}>
                    {s.a}
                    {" "}
                    &harr;
                    {" "}
                    {s.b}
                  </span>
                  <div class={styles.barTrack}>
                    <div class={clsx(styles.barFill, cls.bar)} style={`width: ${Math.max(0, s.score) * 100}%`} />
                  </div>
                  <span class={clsx(styles.similarityScore, cls.text)}>{s.score.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {analogies && analogies.length > 0 && (
        <>
          <div class={styles.label}>Vector Analogies</div>
          {analogies.map((a, i) => (
            <div key={i} class={styles.analogy}>
              <span class={styles.analogyQuery}>{a.query}</span>
              <span>&asymp;</span>
              <span class={styles.analogyResult}>{a.result}</span>
              <span class={styles.analogyScore}>
                (
                {a.score.toFixed(2)}
                )
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
