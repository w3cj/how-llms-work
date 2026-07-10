/** Hook for the from-scratch BPE tokenizer. Streams merge steps as the algorithm builds a vocabulary. */
import type { BpeInit, BpeResult, MergeStep } from "../components/bpe-tokenize-result/index.js";

import { BpeTokenizeResult } from "../components/bpe-tokenize-result/index.js";
import { useSSEChat } from "./use-sse-chat.js";

type BpeEvent = BpeInit | MergeStep | BpeResult;

export function useBpeTokenizeChat() {
  return useSSEChat<{ init?: BpeInit; mergeSteps: MergeStep[]; result?: BpeResult }, BpeEvent>({
    endpoint: "/bpe-tokenize",
    title: "Basic Tokenizer",
    tagline: "watch BPE build a vocabulary from scratch",
    initState: () => ({ mergeSteps: [] }),
    onEvent: (parsed, state) => {
      if ("corpus" in parsed) {
        state.init = parsed as BpeInit;
        return <BpeTokenizeResult init={state.init} mergeSteps={state.mergeSteps} />;
      }
      if ("pair" in parsed) {
        state.mergeSteps.push(parsed as MergeStep);
        return <BpeTokenizeResult init={state.init} mergeSteps={[...state.mergeSteps]} />;
      }
      if ("inputTokens" in parsed) {
        state.result = parsed as BpeResult;
        return <BpeTokenizeResult init={state.init} mergeSteps={[...state.mergeSteps]} result={state.result} />;
      }
    },
  });
}
