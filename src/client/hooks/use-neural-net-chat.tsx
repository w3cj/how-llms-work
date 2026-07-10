/**
 * Hook for XOR neural net training. Input: "single-layer [epochs]" or "multi-layer [epochs]".
 * Streams epoch losses, then shows final predictions and pass/fail verdict.
 */
import type { EpochData, NeuralNetSummary } from "../components/neural-net-result/index.js";

import { NeuralNetResult } from "../components/neural-net-result/index.js";
import { useSSEChat } from "./use-sse-chat.js";

const WHITESPACE = /\s+/;

type NeuralNetEvent = EpochData | NeuralNetSummary;

export function useNeuralNetChat() {
  return useSSEChat<{ epochs: EpochData[] }, NeuralNetEvent>({
    endpoint: "/neural-net",
    title: "Neural Net",
    tagline: "train a neural net on XOR — try single-layer or multi-layer",
    buildBody: (input) => {
      const parts = input.trim().split(WHITESPACE);
      const mode = parts[0] === "multi-layer" ? "multi-layer" : "single-layer";
      const epochs = parts[1] ? Number.parseInt(parts[1], 10) || 5000 : 5000;
      return { mode, epochs };
    },
    initState: () => ({ epochs: [] }),
    onEvent: (parsed, state) => {
      if ("epoch" in parsed) {
        state.epochs.push(parsed as EpochData);
        return <NeuralNetResult epochs={[...state.epochs]} />;
      }
      if ("predictions" in parsed) {
        return <NeuralNetResult epochs={[...state.epochs]} summary={parsed as NeuralNetSummary} />;
      }
    },
  });
}
