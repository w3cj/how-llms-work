/** Request schema for neural net training — single-layer (perceptron) or multi-layer (backprop) mode. */
import { z } from "zod";

export default z.object({
  mode: z.enum(["single-layer", "multi-layer"]),
  epochs: z.number().int().min(100).max(100000).default(5000),
});
