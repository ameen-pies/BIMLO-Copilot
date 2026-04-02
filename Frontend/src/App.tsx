import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Chat from "./pages/Chat";
import NotFound from "./pages/NotFound";
import CallPage from "./pages/CallPage";
import NewsPage from "./pages/NewsPage";
import ClickSpark from "@/components/ClickSpark";
import './scrollbar.css';

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ClickSpark
          sparkColor="#60a5fa"
          sparkSize={8}
          sparkRadius={18}
          sparkCount={8}
          duration={380}
        >
          <div style={{ minHeight: '100vh' }}>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/call" element={<CallPage />} />
              <Route path="/news" element={<NewsPage />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </div>
        </ClickSpark>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;