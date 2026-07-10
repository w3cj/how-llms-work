/** Displays neural network training progress (epoch losses) and final XOR predictions with a pass/fail verdict. Loss values are color-coded: green when low, orange when high. */
import clsx from "clsx";
import styles from "./styles.module.css";

export type EpochData = {
  epoch: number;
  loss: number;
};

export type Prediction = {
  actual: number;
  expected: number;
  input: number[];
};

export type NeuralNetSummary = {
  architecture: string;
  predictions: Prediction[];
  verdict: string;
};

function lossClass(loss: number) {
  if (loss < 0.01)
    return styles.lossLow;
  if (loss > 0.1)
    return styles.lossHigh;
  return "";
}

export function NeuralNetResult({ epochs, summary }: { epochs: EpochData[]; summary?: NeuralNetSummary }) {
  const isSuccess = summary?.verdict.startsWith("SUCCESS");

  return (
    <div class="vstack">
      <div class={styles.label}>
        {summary?.architecture ?? "Training..."}
      </div>

      <div class={styles.epochList}>
        {epochs.map((e, i) => (
          <div key={i} class={styles.epochRow}>
            <span class={styles.epochNum}>
              epoch
              {e.epoch}
            </span>
            <span class={lossClass(e.loss)}>
              loss
              {e.loss.toFixed(6)}
            </span>
          </div>
        ))}
      </div>

      {summary && (
        <>
          <div class={styles.label}>Predictions</div>
          <div class={styles.predictions}>
            {summary.predictions.map((p, i) => {
              const correct = Math.abs(p.actual - p.expected) < 0.1;
              return (
                <div key={i} class={styles.predictionRow}>
                  <span class={styles.predictionInput}>
                    [
                    {p.input.join(", ")}
                    ]
                  </span>
                  <span class={styles.predictionExpected}>
                    expected
                    {p.expected}
                  </span>
                  <span>→</span>
                  <span class={correct ? styles.correct : styles.incorrect}>{p.actual.toFixed(2)}</span>
                </div>
              );
            })}
          </div>

          <div class={clsx(styles.verdict, isSuccess ? styles.verdictSuccess : styles.verdictFailed)}>
            {summary.verdict}
          </div>
        </>
      )}
    </div>
  );
}
