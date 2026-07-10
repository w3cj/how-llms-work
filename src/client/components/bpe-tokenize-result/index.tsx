/** Displays BPE tokenization results: training text, initial characters, merge steps (pair → merged token with frequency), learned vocabulary, and final tokenized output with compression ratio. Each section is collapsible. */
import styles from "./styles.module.css";

export type BpeInit = {
  corpus: string;
  characters: string[];
  charCount: number;
  wordCount: number;
};

export type MergeStep = {
  step: number;
  pair: [string, string];
  frequency: number;
  newToken: string;
  vocabSize: number;
  tokenCount: number;
};

export type BpeResult = {
  inputTokens: string[];
  tokenCount: number;
  originalCharCount: number;
  compressionRatio: string;
};

function displayToken(t: string): string {
  return t.replaceAll(" ", "\u2423").replaceAll("\n", "\\n").replaceAll("\t", "\\t");
}

export function BpeTokenizeResult({
  init,
  mergeSteps,
  result,
}: {
  init?: BpeInit;
  mergeSteps: MergeStep[];
  result?: BpeResult;
}) {
  return (
    <div class="vstack">
      {init && (
        <>
          <details open class={styles.section}>
            <summary class={styles.label}>
              Pre-tokenized (
              {init.wordCount}
              {" "}
              unique words)
            </summary>
            <pre class={styles.corpus}>{init.corpus}</pre>
          </details>

          <details open class={styles.section}>
            <summary class={styles.label}>
              Characters (
              {init.charCount}
              )
            </summary>
            <div class={styles.tokens}>
              {init.characters.map((ch, i) => (
                <span key={i} class={`badge outline ${styles.charBadge}`}>
                  {displayToken(ch)}
                </span>
              ))}
              {init.charCount > init.characters.length && (
                <span class={styles.truncated}>
                  +
                  {init.charCount - init.characters.length}
                  {" "}
                  more
                </span>
              )}
            </div>
          </details>
        </>
      )}

      {mergeSteps.length > 0 && (
        <details open class={styles.section}>
          <summary class={styles.label}>
            Merge Steps (
            {mergeSteps.length}
            )
          </summary>
          <div class={styles.mergeList}>
            {mergeSteps.map((m, i) => (
              <div key={i} class={styles.mergeRow}>
                <span class={styles.stepNum}>
                  {m.step}
                  .
                </span>
                <span class={styles.pair}>
                  <span class={styles.pairToken}>{displayToken(m.pair[0])}</span>
                  {" + "}
                  <span class={styles.pairToken}>{displayToken(m.pair[1])}</span>
                </span>
                <span class={styles.arrow}>{"\u2192"}</span>
                <span class={styles.merged}>{displayToken(m.newToken)}</span>
                <span class={styles.freq}>
                  {"\u00D7"}
                  {m.frequency}
                </span>
                <span class={styles.stats}>
                  vocab
                  {m.vocabSize}
                  {" "}
                  | tokens
                  {m.tokenCount}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}

      {result && (
        <>
          <details open class={styles.section}>
            <summary class={styles.label}>
              Vocabulary (
              {new Set(result.inputTokens).size}
              {" "}
              unique tokens)
            </summary>
            <div class={styles.tokens}>
              {[...new Set(result.inputTokens)].map(t => (
                <span key={t} class={`badge outline ${styles.vocabBadge}`}>
                  {displayToken(t)}
                </span>
              ))}
            </div>
          </details>

          <details open class={styles.section}>
            <summary class={styles.label}>Your Text, Tokenized</summary>
            <div class={styles.tokens}>
              {result.inputTokens.map((t, i) => (
                <span key={i} class={`badge outline ${styles.resultBadge}`}>
                  {displayToken(t)}
                </span>
              ))}
            </div>
          </details>

          <div class={styles.compression}>
            {result.originalCharCount}
            {" "}
            chars
            {"\u2192"}
            {" "}
            {result.tokenCount}
            {" "}
            tokens (
            {result.compressionRatio}
            {" "}
            compression)
          </div>
        </>
      )}
    </div>
  );
}
