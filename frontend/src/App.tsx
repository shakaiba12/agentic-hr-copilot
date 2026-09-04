import { useEffect, useState } from 'react';
import { ChatPage } from './components/ChatPage';
import { useChat } from './hooks/useChat';
import { fetchHealth, fetchDocuments } from './services/api';
import { HealthStatus, PolicyDocument } from './types/api';

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [documents, setDocuments] = useState<PolicyDocument[]>([]);

  const {
    messages,
    loading,
    streaming,
    error,
    sendMessage,
    clearChat,
    stopGeneration,
  } = useChat();

  useEffect(() => {
    let isMounted = true;

    async function loadInitialData() {
      try {
        const [healthData, docsData] = await Promise.allSettled([
          fetchHealth(),
          fetchDocuments(),
        ]);

        if (isMounted) {
          if (healthData.status === 'fulfilled') {
            setHealth(healthData.value);
          }
          if (docsData.status === 'fulfilled') {
            setDocuments(docsData.value);
          }
        }
      } catch {
        // Handled silently
      }
    }

    loadInitialData();

    // Periodic health check every 30 seconds
    const interval = setInterval(async () => {
      try {
        const h = await fetchHealth();
        if (isMounted) setHealth(h);
      } catch {
        // Ignore
      }
    }, 30000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <ChatPage
      messages={messages}
      loading={loading}
      streaming={streaming}
      error={error}
      health={health}
      documents={documents}
      onSendMessage={sendMessage}
      onClearChat={clearChat}
      onStopGeneration={stopGeneration}
    />
  );
}

export default App;
