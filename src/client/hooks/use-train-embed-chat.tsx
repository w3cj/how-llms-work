/**
 * Hook for Word2Vec Skip-gram training. Input is comma- or space-separated words to compare.
 * Streams corpus stats, epoch losses, then learned embeddings with neighbors and similarities.
 */
import type { Analogy, EpochData, InitData, Neighbor, SimilarityPair, WordEmbedding } from "../components/train-embed-result/index.js";

import { TrainEmbedResult } from "../components/train-embed-result/index.js";
import { useSSEChat } from "./use-sse-chat.js";

const WHITESPACE = /\s+/;

type TrainEmbedState = {
  init?: InitData;
  epochs: EpochData[];
  embeddings?: WordEmbedding[];
  neighbors?: Neighbor[];
  similarities?: SimilarityPair[];
  analogies?: Analogy[];
  warnings?: string[];
};

type DoneEvent = {
  embeddings: WordEmbedding[];
  neighbors: Neighbor[];
  similarities: SimilarityPair[];
  analogies: Analogy[];
  warnings: string[];
};

type TrainEmbedEvent = InitData | EpochData | DoneEvent;

export function useTrainEmbedChat() {
  return useSSEChat<TrainEmbedState, TrainEmbedEvent>({
    endpoint: "/train-embed",
    title: "Train Embeddings",
    tagline: "train word2vec skip-gram from scratch — enter words to compare",
    buildBody: (input) => {
      const words = (input.includes(",") ? input.split(",") : input.split(WHITESPACE))
        .map(w => w.trim().toLowerCase())
        .filter(Boolean);
      return { words };
    },
    initState: () => ({ epochs: [] }),
    onEvent: (parsed, state) => {
      if ("vocabSize" in parsed) {
        state.init = parsed as InitData;
        return <TrainEmbedResult init={state.init} epochs={[]} />;
      }
      if ("epoch" in parsed) {
        state.epochs.push(parsed as EpochData);
        return <TrainEmbedResult init={state.init} epochs={[...state.epochs]} />;
      }
      if ("embeddings" in parsed) {
        const done = parsed as DoneEvent;
        state.embeddings = done.embeddings;
        state.neighbors = done.neighbors;
        state.similarities = done.similarities;
        state.analogies = done.analogies;
        state.warnings = done.warnings;
        return (
          <TrainEmbedResult
            init={state.init}
            epochs={[...state.epochs]}
            embeddings={state.embeddings}
            neighbors={state.neighbors}
            similarities={state.similarities}
            analogies={state.analogies}
            warnings={state.warnings}
          />
        );
      }
    },
  });
}
