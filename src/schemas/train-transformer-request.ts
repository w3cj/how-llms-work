/** Request schema for transformer training — epochs, generation settings (temperature, top-p), and architecture (layers, max tokens). */
import { z } from "zod";

export default z.object({
  epochs: z.number().int().min(50).max(2000).default(300),
  temperature: z.number().min(0.1).max(2.0).default(0.8),
  topP: z.number().min(0.1).max(1.0).default(0.9),
  numLayers: z.number().int().min(1).max(6).default(2),
  maxTokens: z.number().int().min(3).max(500).default(40),
});
