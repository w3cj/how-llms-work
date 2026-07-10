/** Root app layout — wraps a page's chat hook with the header, message list, and chat input. */
import type { JSX } from "hono/jsx/jsx-runtime";

import type { ChatState } from "../../context/chat-context.js";
import { ChatProvider } from "../../context/chat-provider.js";
import { useAutoScroll } from "../../hooks/use-auto-scroll.js";
import { ChatInput } from "../chat-input/index.js";
import { Header } from "../header/index.js";
import { MessageList } from "../message-list/index.js";
import styles from "./styles.module.css";

export function App({ chat, slots }: { chat: ChatState; slots?: { belowHeader?: JSX.Element; aboveChat?: JSX.Element } }) {
  const {
    loading,
    messages,
    sendMessage,
  } = chat;
  const {
    ref: chatRef,
    handleScroll,
    scrollToBottom,
  } = useAutoScroll([messages, loading]);

  const handleSend = () => {
    scrollToBottom();
    sendMessage();
  };

  return (
    <ChatProvider value={chat}>
      <div class={styles.app}>
        <Header />
        {slots?.belowHeader}
        <MessageList onScroll={handleScroll} scrollRef={chatRef} />
        {slots?.aboveChat}
        <ChatInput onSend={handleSend} />
      </div>
    </ChatProvider>
  );
}
