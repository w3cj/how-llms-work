/**
 * Hook for transformer training. Input: epoch count (or just press send for default 300).
 * Streams architecture stats, epoch losses with text generation samples, then final results.
 */
import type { EpochData, InitData, Sample, TransformerSummary } from "../components/train-transformer-result/index.js";

import { TrainTransformerResult } from "../components/train-transformer-result/index.js";
import { useSSEChat } from "./use-sse-chat.js";

const WHITESPACE = /\s+/;

type TrainTransformerState = {
  init?: InitData;
  epochs: EpochData[];
  samples: Sample[];
  summary?: TransformerSummary;
};

type DoneEvent = {
  architecture: string;
  finalLoss: number;
  samples: Sample[];
};

type TrainTransformerEvent = InitData | EpochData | DoneEvent;

export function useTrainTransformerChat() {
  return useSSEChat<TrainTransformerState, TrainTransformerEvent>({
    endpoint: "/train-transformer",
    title: "Train Transformer",
    tagline: "train a GPT from scratch — try: 300 0.8 0.9 2 40 (epochs, temp, top-p, layers, max tokens)",
    buildBody: (input) => {
      const parts = input.trim().split(WHITESPACE);
      const epochs = Number.parseInt(parts[0], 10) || 300;
      const temperature = parts[1] ? Number.parseFloat(parts[1]) || 0.8 : 0.8;
      const topP = parts[2] ? Number.parseFloat(parts[2]) || 0.9 : 0.9;
      const numLayers = parts[3] ? Number.parseInt(parts[3], 10) || 2 : 2;
      const maxTokens = parts[4] ? Number.parseInt(parts[4], 10) || 40 : 40;
      return { epochs, temperature, topP, numLayers, maxTokens };
    },
    initState: () => ({ epochs: [], samples: [] }),
    onEvent: (parsed, state) => {
      if ("vocabSize" in parsed && "totalParams" in parsed) {
        state.init = parsed as InitData;
        return <TrainTransformerResult init={state.init} epochs={[]} samples={[]} />;
      }
      if ("epoch" in parsed) {
        const ep = parsed as EpochData;
        state.epochs.push(ep);
        if (ep.sample)
          state.samples.push({ epoch: ep.epoch, text: ep.sample });
        return (
          <TrainTransformerResult
            init={state.init}
            epochs={[...state.epochs]}
            samples={[...state.samples]}
          />
        );
      }
      if ("architecture" in parsed) {
        const done = parsed as DoneEvent;
        state.summary = { architecture: done.architecture, finalLoss: done.finalLoss };
        state.samples = done.samples;
        return (
          <TrainTransformerResult
            init={state.init}
            epochs={[...state.epochs]}
            samples={[...state.samples]}
            summary={state.summary}
          />
        );
      }
    },
  });
}
