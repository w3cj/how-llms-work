/** Client entry point — mounts the app into the #root element. */
import { render } from "hono/jsx/dom";

import { Root } from "./root.js";
import "./styles.css";

const root = document.querySelector<HTMLElement>("#root");

if (root) {
  render(<Root />, root);
}
