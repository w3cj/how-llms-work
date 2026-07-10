/** Displays transformer training progress: architecture stats, epoch losses, generated text samples at different training stages, and final result. */
import styles from "./styles.module.css";

export type InitData = {
  vocabSize: number;
  contextLen: number;
  embeddingDim: number;
  numHeads: number;
  ffDim: number;
  numLayers: number;
  totalParams: number;
  temperature: number;
  topP: number;
  corpusSentences: number;
  trainingSequences: number;
};

export type EpochData = {
  epoch: number;
  loss: number;
  sample?: string;
};

export type Sample = {
  epoch: number;
  text: string;
};

export type TransformerSummary = {
  architecture: string;
  finalLoss: number;
};

type Props = {
  init?: InitData;
  epochs: EpochData[];
  samples: Sample[];
  summary?: TransformerSummary;
};

function lossClass(loss: number) {
  if (loss < 2.0)
    return styles.lossLow;
  if (loss > 4.0)
    return styles.lossHigh;
  return "";
}

export function TrainTransformerResult({ init, epochs, samples, summary }: Props) {
  return (
    <div class="vstack">
      {init && (
        <>
          <div class={styles.label}>Architecture</div>
          <div class={styles.config}>
            <div class={styles.configItem}>
              vocab
              {" "}
              <span class={styles.configValue}>{init.vocabSize}</span>
            </div>
            <div class={styles.configItem}>
              embedding
              {" "}
              <span class={styles.configValue}>{init.embeddingDim}</span>
            </div>
            <div class={styles.configItem}>
              layers
              {" "}
              <span class={styles.configValue}>{init.numLayers}</span>
            </div>
            <div class={styles.configItem}>
              heads
              {" "}
              <span class={styles.configValue}>{init.numHeads}</span>
            </div>
            <div class={styles.configItem}>
              ff hidden
              {" "}
              <span class={styles.configValue}>{init.ffDim}</span>
            </div>
            <div class={styles.configItem}>
              context
              {" "}
              <span class={styles.configValue}>{init.contextLen}</span>
            </div>
            <div class={styles.configItem}>
              parameters
              {" "}
              <span class={styles.configValue}>{init.totalParams.toLocaleString()}</span>
            </div>
            <div class={styles.configItem}>
              temperature
              {" "}
              <span class={styles.configValue}>{init.temperature}</span>
            </div>
            <div class={styles.configItem}>
              top-p
              {" "}
              <span class={styles.configValue}>{init.topP}</span>
            </div>
            <div class={styles.configItem}>
              sequences
              {" "}
              <span class={styles.configValue}>{init.trainingSequences}</span>
            </div>
          </div>
        </>
      )}

      {epochs.length > 0 && (
        <>
          <div class={styles.label}>{summary ? "Training" : "Training..."}</div>
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

      {samples.length > 0 && (
        <>
          <div class={styles.label}>Generated Text</div>
          <div class={styles.samples}>
            {samples.map((s, i) => (
              <div key={i} class={styles.sample}>
                <div class={styles.sampleEpoch}>
                  epoch
                  {" "}
                  {s.epoch}
                </div>
                <div class={styles.sampleText}>{s.text}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {summary && (
        <div class={styles.verdict}>
          {summary.architecture}
          {" "}
          — final loss
          {" "}
          {summary.finalLoss.toFixed(4)}
        </div>
      )}
    </div>
  );
}
