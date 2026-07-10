/** Request schema for embedding training — query words to search after training, plus Skip-gram hyperparameters. */
import { z } from "zod";

export default z.object({
  words: z.array(z.string().min(1)).min(1).max(10),
  epochs: z.number().int().min(10).max(10000).default(10000),
  dimensions: z.number().int().min(4).max(64).default(32),
  windowSize: z.number().int().min(1).max(5).default(2),
  negativeSamples: z.number().int().min(1).max(10).default(5),
});
